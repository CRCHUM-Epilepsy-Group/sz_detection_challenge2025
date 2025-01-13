from szdetect import model as mod

models = mod.parse_models("models.yaml")

models_inst = mod.init_models(models['models'], mod.Model)
