import torch
import pickle


def save_var(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()
    return filename


def EnRML(pred_enn, params, initial, lamuda, dstb, error_per):
    '''
    print("EnRML input: \n",
          "pred_enn: {}\nparams: {}\ninitial: {}\nlamuda: {} \ndstb: {}\nerror_per:{}".format(
              pred_enn.shape, params.shape, initial.shape, lamuda, dstb.shape, error_per
          ))

    Input: x and y
    Trainable parameter: m
    Hyper-parameters: mpr , CD and CM (determined based on prior information)
    For j = 1,...,Ne
    1. Generate realizations of measurement error  based on its probability distribution function (PDF);
    2. Generate initial realizations of the model parameters mj based on prior PDF;
    3. Calculate the observed data dobs by adding the measurement error  to the target value y ;
    repeat
        Step 1: Compute the predicted data g(mj ) for each realization based on the model parameters;
        Step 2: Update the model parameters mj according to Eq. (10). The CMl ,Dl and CDl ,Dl are
        calculated among the ensemble of realizations. Thus, the ensemble of realizations is updated
        simultaneously;
    until the training loss has converged;
    '''
    pred_enn = pred_enn.cpu()
    NE = pred_enn.shape[0]
    data = pred_enn.reshape(NE, -1).t()
    Cd = (torch.eye(data.shape[0]) * (error_per**2))
    dparams = params - torch.stack([params.mean(1)]*NE).t()
    ddata = data - torch.stack([data.mean(1)]*NE).t()
    C_md = torch.mm(dparams, ddata.t())/(NE-1)
    C_dd = torch.mm(ddata, ddata.t())/(NE-1)
    C_mm = torch.mm(dparams, dparams.t())/(NE-1)
    C_m = torch.eye(params.size()[0])
    # the cuda calculatin begins
    with torch.no_grad():
        t0 = ((1+lamuda)*Cd + C_dd).cuda()
        t1 = t0.inverse()
        a = torch.mm(t1, C_md.t().cuda())
        b = torch.mm(C_md.cuda(), a)
        c = torch.mm(C_mm.cuda() - b, C_m.cuda().inverse())
        delta_m = -(1/(1+lamuda)) * torch.mm(c, (params - initial).cuda())
        delta_gm = -torch.mm(C_md.cuda(), torch.mm(t1, (data.cuda() - dstb)))
        delta = (delta_m + delta_gm).cpu()
        # del(pred_enn, params, initial, NE, Cd,
        # dparams, ddata, C_md, C_dd, C_mm, C_m, t1, a, b, c, delta_m, delta_gm, lamuda)
    torch.cuda.empty_cache()
    return delta
