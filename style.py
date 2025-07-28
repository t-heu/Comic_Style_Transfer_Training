import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from src.utils import save_img, get_img, exists, list_files
import numpy as np
import evaluate
from optimize import optimize

CONTENT_WEIGHT = 1e1
STYLE_WEIGHT = 1e4
TV_WEIGHT = 1e-6

LEARNING_RATE = 5e-4
NUM_EPOCHS = 10
CHECKPOINT_DIR = 'checkpoints'
CHECKPOINT_ITERATIONS = 2000
VGG_PATH = 'data/imagenet-vgg-verydeep-19.mat'
TRAIN_PATH = 'data/train2014'  # 'data/train2014'
BATCH_SIZE = 8

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-dir', type=str,
                        dest='checkpoint_dir', help='dir to save checkpoint in',
                        metavar='CHECKPOINT_DIR', required=True)

    parser.add_argument('--style', type=str,
                        dest='style', help='style image path',
                        metavar='STYLE', required=True)

    parser.add_argument('--train-path', type=str,
                        dest='train_path', help='path to training images folder',
                        metavar='TRAIN_PATH', default=TRAIN_PATH)

    parser.add_argument('--test', type=str,
                        dest='test', help='test image path',
                        metavar='TEST', default=None)

    parser.add_argument('--test-dir', type=str,
                        dest='test_dir', help='test image save dir',
                        metavar='TEST_DIR', default=None)

    parser.add_argument('--slow', dest='slow', action='store_true',
                        help='gatys approach (for debugging, not supported)',
                        default=False)

    parser.add_argument('--epochs', type=int,
                        dest='epochs', help='num epochs',
                        metavar='EPOCHS', default=NUM_EPOCHS)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size', help='batch size',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)

    parser.add_argument('--checkpoint-iterations', type=int,
                        dest='checkpoint_iterations', help='checkpoint frequency',
                        metavar='CHECKPOINT_ITERATIONS',
                        default=CHECKPOINT_ITERATIONS)

    parser.add_argument('--vgg-path', type=str,
                        dest='vgg_path',
                        help='path to VGG19 network',
                        metavar='VGG_PATH', default=VGG_PATH)

    parser.add_argument('--content-weight', type=float,
                        dest='content_weight',
                        help='content weight',
                        metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)

    parser.add_argument('--style-weight', type=float,
                        dest='style_weight',
                        help='style weight',
                        metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)

    parser.add_argument('--tv-weight', type=float,
                        dest='tv_weight',
                        help='total variation regularization weight',
                        metavar='TV_WEIGHT', default=TV_WEIGHT)

    parser.add_argument('--learning-rate', type=float,
                        dest='learning_rate',
                        help='learning rate',
                        metavar='LEARNING_RATE', default=LEARNING_RATE)

    return parser

def check_opts(opts):
    exists(opts.checkpoint_dir, "checkpoint dir not found!")
    exists(opts.style, "style path not found!")
    exists(opts.train_path, "train path not found!")
    if opts.test or opts.test_dir:
        exists(opts.test, "test img not found!")
        exists(opts.test_dir, "test directory not found!")
    exists(opts.vgg_path, "vgg network data not found!")
    assert opts.epochs > 0
    assert opts.batch_size > 0
    assert opts.checkpoint_iterations > 0
    assert os.path.exists(opts.vgg_path)
    assert opts.content_weight >= 0
    assert opts.style_weight >= 0
    assert opts.tv_weight >= 0
    assert opts.learning_rate >= 0

def _get_files(img_dir):
    files = list_files(img_dir)
    return [os.path.join(img_dir, x) for x in files]

def main():
    parser = build_parser()
    options = parser.parse_args()
    check_opts(options)

    style_target = get_img(options.style)
    if not options.slow:
        content_targets = _get_files(options.train_path)
    elif options.test:
        content_targets = [options.test]
    else:
        content_targets = []

    kwargs = {
        "slow": options.slow,
        "epochs": options.epochs,
        "print_iterations": options.checkpoint_iterations,
        "batch_size": options.batch_size,
        "save_path": os.path.join(options.checkpoint_dir, 'fns.ckpt'),
        "learning_rate": options.learning_rate
    }

    if options.slow:
        if options.epochs < 10:
            kwargs['epochs'] = 1000
        if options.learning_rate < 1:
            kwargs['learning_rate'] = 1e1

    args = [
        content_targets,
        style_target,
        options.content_weight,
        options.style_weight,
        options.tv_weight,
        options.vgg_path
    ]

    for preds, losses, i, epoch in optimize(*args, **kwargs):
        style_loss, content_loss, tv_loss, loss = losses

        print(f'Epoch {epoch}, Iteration: {i}, Loss: {loss:.6f}')
        total_batches = len(content_targets) // options.batch_size
        total_steps = total_batches * options.epochs
        current_step = epoch * total_batches + i
        progress = (current_step / total_steps) * 100
        print(f"[PROGRESS] {current_step}/{total_steps} steps — {progress:.2f}% done\n")

        if options.test and options.test_dir:
            preds_path = f'{options.test_dir}/{epoch}_{i}.png'
            if not options.slow:
                evaluate.ffwd_to_img(options.test, preds_path, options.checkpoint_dir)
            else:
                save_img(preds_path, preds)

    print("\n✅ Training finished successfully!")
    print(f"📍Model saved in: {options.checkpoint_dir}")
    print(f"📸 To test, run: python evaluate.py --checkpoint {options.checkpoint_dir} --in-path test_images --out-path results")


if __name__ == '__main__':
    main()
