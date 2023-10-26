import torch
import torch.nn as nn
from torchattacks.attack import Attack


class WarmupPGD(Attack):
    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels, init_delta=None):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        if self.random_start:
            adv_images = images.clone().detach()
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        elif init_delta is not None:
            adv_images = images.clone().detach() + init_delta
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        else:
            assert False

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


def pgd_attack_classifier(model, eps, alpha, steps, images, labels, init_delta=None):
    # Check input types, here we assume PIL images, imagenet normalization
    assert isinstance(images, torch.Tensor)

    # Create an instance of the attack
    attack = WarmupPGD(
        model,
        eps=eps,
        alpha=alpha,
        steps=steps,
        random_start=True if init_delta is None else False,
    )

    # ImageNet normalization
    attack.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # Set targeted mode
    attack.set_mode_targeted_by_label(quiet=True)

    # Perform the attack, always attack to non-watermark class
    target_labels = torch.zeros_like(labels, dtype=torch.long).to(labels.device)
    images_adv = attack(images, target_labels, init_delta=init_delta)

    return images_adv
