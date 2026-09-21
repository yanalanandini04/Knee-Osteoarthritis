import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.forward_hook = target_layer.register_forward_hook(self._save_activations)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _, __, output):
        self.activations = output

    def _save_gradients(self, _, __, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, image, class_index=None):
        self.model.zero_grad(set_to_none=True)
        output = self.model(image)
        class_index = int(output.argmax(1).item()) if class_index is None else class_index
        output[:, class_index].sum().backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True).clamp(min=0)
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach().cpu().numpy(), class_index

    def close(self):
        self.forward_hook.remove()
        self.backward_hook.remove()
