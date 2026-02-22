import torch
import torch.nn.functional as F
import math
from model import CrossModalCLIPModel
from train import VOCAB_SIZE, get_static_4_item_batch

def evaluate_cross_modal():
    """
    Evaluates the Zero-Shot Retrieval capability of the Cross-Modal model.
    We ask the model to find the correct Price Chart given a specific News Event text.
    """
    print("Evaluating Cross-Modal Zero-Shot Retrieval...")
    
    # Load model
    model = CrossModalCLIPModel(vocab_size=VOCAB_SIZE, projection_dim=32)
    model.load_state_dict(torch.load("cross_modal_clip.pth", weights_only=True))
    model.eval()
    
    # Generate the validation 4-item static batch
    gallery_price, gallery_text = get_static_4_item_batch(window_size=128)
    
    with torch.no_grad():
        v_price, v_text = model(gallery_price, gallery_text)
        
        # Calculate similarity matrix (-1 to 1) 
        sim_matrix = v_text @ v_price.T
        
        correct_top1 = 0
        correct_top2 = 0
        
        for i in range(4):
            similarities = sim_matrix[i]
            top_indices = torch.argsort(similarities, descending=True)
            
            if top_indices[0] == i:
                correct_top1 += 1
            if i in top_indices[:2]:
                correct_top2 += 1
                
        print(f"Text -> Chart Retrieval Recall@1:  {correct_top1 / 4 * 100}%")
        print(f"Text -> Chart Retrieval Recall@2:  {correct_top2 / 4 * 100}%")
        
        if correct_top1 / 4 > 0.8:
            print("RESULT: SUCCESS - The model successfully learned to cross-translate modalities.")
        else:
            print("RESULT: WARNING - Low retrieval accuracy. Modalities are not aligned.")

if __name__ == "__main__":
    evaluate_cross_modal()
