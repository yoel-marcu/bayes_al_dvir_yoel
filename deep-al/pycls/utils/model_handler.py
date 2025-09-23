import pycls.core.builders as model_builder
import pycls.utils.checkpoint as cu

def get_model(cfg, ckpt_file=None, model_init_state=None):
    model = model_builder.build_model(cfg)
    if ckpt_file is not None:
        model = cu.load_checkpoint(ckpt_file, model, weights_only=False)
    if model_init_state is not None and not cfg.MODEL.USE_1NN:
        model.load_state_dict(model_init_state)
    return model