from ..builder import PIPELINES


@PIPELINES.register_module()
class ToZeroOneScale(object):
    def __call__(self, results):
        results['img'] = results['img'] / 255
        return results
