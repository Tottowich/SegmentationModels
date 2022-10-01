from transformers import ViTModel, ViTConfig

if __name__=="__main__":
    config = ViTConfig()
    model = ViTModel(config)
    print(model)