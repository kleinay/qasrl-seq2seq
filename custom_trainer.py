from transformers import Seq2SeqTrainer

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def compute_loss(self, model, inputs, **kwargs):
        ret = super().compute_loss(model, inputs, **kwargs)
        return ret
    
    def training_step(self, model, inputs):
        ret = super().compute_loss(model, inputs)
        return ret
        