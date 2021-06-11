from torchmeta.modules import MetaModule


class MetaModuleMonteCarlo(MetaModule):
    """
    For Monte Carlo
    """
    def meta_named_parameters(self, prefix='', recurse=True):
        gen = self._named_members(
            lambda module: module._parameters.items()
            if isinstance(module, MetaModuleMonteCarlo) else [],
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem
