# Zoltraak

**Zero-dependency, single-file GPT inference in pure Java.**

Zoltraak is a bare-metal implementation of a Transformer language model heavily inspired by karpathy/llama2.c. It strips away the abstractions of PyTorch and HuggingFace to show the raw arithmetic of Large Language Models. There are no external libraries, no ND4J, no numpy, just raw float[] arrays, manual memory offsets, and a custom implementation of Matrix Multiplication and Byte Pair Encoding running on the JVM.

## Usage

### 1. Train & Export

You'll need Python and PyTorch for the training leg. The script downloads data, trains the model, and exports the weights to our custom binary format (.bin).

```bash
# Install dependencies
pip install torch requests

# Run training (produces model.bin and tokenizer.bin)
# Use 'fast' for a quick smoke test or 'colab' for actual training
python train_gpt.py --mode fast
```

2. Compile & Run

Switch to Java for the inference runtime.

```bash
javac ZoltraakTLMvf.java
java ZoltraakTLMvf
```

# Train Your Own Model

The default parameters in train_gpt.py (under mode='fast') are intentionally crippled for speed: it uses a tiny embedding size and practically no layers, so don't expect it to write Shakespeare. It will mostly output incoherent babble because it's just testing the plumbing.

To get a model that actually speaks English:

1. Open train_gpt.py.

2. Modify the Config class.

3. Switch to the colab settings (or crank them higher if you have the VRAM).

4. Increase max_iters.

5. Let it cook on a GPU for a few hours.

# Why Java?

Because it was an assignment.


# Development Methodology

This project was built using an AI-Assisted Workflow. I utilized LLMs to generate verbose boilerplate code and handle syntax translation between Python and Java. This allowed me to focus entirely on the system architecture, the binary specification, and the critical debugging required to align the floating-point precision of the Java tensor operations with the PyTorch reference implementation.
