import torch
import numpy as np
from datasets import load_metric
from torchvision.transforms import Compose, ToTensor, Normalize, RandomResizedCrop, \
    RandomHorizontalFlip, CenterCrop, Resize, ToPILImage
from transformers import Trainer, TrainingArguments, ViTFeatureExtractor, ViTForImageClassification
from load_yoga_dataset import YogaPoseDatasetTransformer


def preprocess_train(sample):
    image = train_transforms(sample['image'])
    pixel_values = sample['image'].float()

    # Convert the tensor to a PIL image
    if pixel_values.shape[0] == 1:
        pixel_values = torch.cat([pixel_values, pixel_values, pixel_values], dim=0)

    return {'image': image, 'label': sample['label'], 'pixel_values': pixel_values}


def preprocess_val(sample):
    image = val_transforms(sample['image'])
    pixel_values = sample['image'].float()

    # Convert the tensor to a PIL image
    if pixel_values.shape[0] == 1:
        pixel_values = torch.cat([pixel_values, pixel_values, pixel_values], dim=0)

    return {'image': image, 'label': sample['label'], 'pixel_values': pixel_values}


def compute_metrics(p):
    metric = load_metric("accuracy")
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


def collate_fn(batch):
    # Resize all images and pixel_values to a common size
    for i, sample in enumerate(batch):
        batch[i]['image'] = Resize((224, 224))(sample['image'])
        batch[i]['pixel_values'] = Resize((224, 224))(sample['pixel_values'])

    labels = torch.tensor([x['label'] for x in batch])
    # Stack pixel_values into a tensor
    pixel_values = torch.stack([x['pixel_values'] for x in batch], dim=0)

    return {'pixel_values': pixel_values, 'labels': labels}


train_file = '../dataset/Yoga-82/yoga_train83.txt'
test_file = '../dataset/Yoga-82/yoga_test83.txt'
model_name_or_path = 'google/vit-base-patch16-224-in21k'

# Get transformations from model's feature_extractor
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)
normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
train_transforms = Compose([
    ToPILImage(),
    RandomResizedCrop((feature_extractor.size['height'], feature_extractor.size['width'])),
    RandomHorizontalFlip(),
    ToTensor(),
    normalize,
])
val_transforms = Compose([
    ToPILImage(),
    Resize((feature_extractor.size['height'], feature_extractor.size['width'])),
    CenterCrop((feature_extractor.size['height'], feature_extractor.size['width'])),
    ToTensor(),
    normalize,
])

# Create dataset instances
train_dataset = YogaPoseDatasetTransformer(train_file, preprocess_train)
val_dataset = YogaPoseDatasetTransformer(test_file, preprocess_val)

model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=83,
    ignore_mismatched_sizes=True
)

# Training (all the values here are taken from the official documentation)
training_args = TrainingArguments(
    '../models/vit_test4',
    per_device_train_batch_size=32,
    evaluation_strategy="steps",
    num_train_epochs=6,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=2e-3,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to='tensorboard',
    load_best_model_at_end=True,
)

trainer = Trainer(
    model,
    training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

train_results = trainer.train()

trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

# Evaluation
metrics = trainer.evaluate()
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)
