import os.path as osp
pkg_dir = osp.abspath(osp.dirname(__file__))
data_dir = osp.join(pkg_dir, 'data')
__version__ = '0.1'
__all__ = [
    'enn',
    'ennloss',
    'enrml',
    'lamuda',
]