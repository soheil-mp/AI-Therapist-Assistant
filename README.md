# 🧠 AI Therapist Assistant

Fine-tuned Llama-3.2 1B offering mental health conversations through a secure, local-first approach.

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Technical Requirements](#technical-requirements)
- [Dataset](#dataset)
- [Installation](#installation)
- [Training Process](#training-process)
- [Usage](#usage)
- [Ethical Considerations](#ethical-considerations)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project aims to create an AI-powered therapist assistant by fine-tuning the Llama-3.2 1B model. The assistant is designed to provide initial mental health support while clearly communicating its limitations and directing users to professional help when needed. The model is optimized to run efficiently on consumer-grade GPUs while maintaining high-quality therapeutic responses.

## ✨ Key Features

- Empathetic responses based on CBT (Cognitive Behavioral Therapy) principles
- Privacy-focused design with local deployment options
- Clear communication of AI limitations and scope
- Professional help referral system
- Lightweight model optimized for consumer GPUs
- Quantized model support for efficient inference
- Open-source and customizable
- Regular updates and community contributions

## 💻 Technical Requirements

- NVIDIA GPU with 6GB VRAM (RTX 3060 recommended)
- 32GB RAM
- Ubuntu 20.04+ or Windows 10/11 with WSL2
- Python 3.9+
- PyTorch 2.0+
- 50GB free disk space
- CUDA 11.7+

## 📊 Dataset

Our training utilizes a carefully curated combination of high-quality datasets:

1. **Mental Health Conversations Dataset**
   - Source: [Mental Health Conversations Dataset](https://huggingface.co/datasets/mental-health-conversations)
   - ~15,000 therapeutic conversations
   - Focused on various mental health topics
   - Cleaned and annotated for quality

2. **Therapy Transcripts and Dialogues**
   - Source: [PsyQA Dataset](https://github.com/Princeton-SysML/PsyQA)
   - ~10,000 therapy conversation excerpts
   - Professionally validated responses
   - Privacy-compliant and anonymized

3. **CBT Framework Dataset**
   - Custom-curated dataset of CBT techniques
   - ~3,000 examples of CBT-based interventions
   - Includes common therapeutic scenarios
   - Validated by mental health professionals

4. **Crisis Response Dataset**
   - Specialized dataset for crisis intervention
   - ~2,000 examples of appropriate crisis responses
   - Clear escalation protocols
   - Professional referral guidelines 

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-therapist-assistant.git
   cd ai-therapist-assistant
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate  # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the pre-trained model:
   ```bash
   python scripts/download_model.py
   ```

## 🔄 Training Process

1. **Data Preparation**
   ```bash
   python scripts/prepare_data.py --dataset_dir data/raw --output_dir data/processed
   ```

2. **Model Fine-tuning**
   ```bash
   python scripts/train.py \
       --model_name "llama-3.2-1b" \
       --dataset_path "data/processed" \
       --output_dir "models/fine_tuned" \
       --batch_size 4 \
       --gradient_accumulation_steps 4 \
       --learning_rate 2e-5
   ```

3. **Model Quantization**
   ```bash
   python scripts/quantize.py --model_path "models/fine_tuned"
   ```

## 💡 Usage

1. Start the assistant:
   ```bash
   python run_assistant.py --model_path "models/fine_tuned"
   ```

2. Access the web interface:
   ```
   http://localhost:8080
   ```

## 🤝 Ethical Considerations

- Not a replacement for professional mental health care
- Clear disclosure of AI nature
- Regular bias monitoring and mitigation
- Privacy-first approach with local deployment options
- Transparent referral system to mental health professionals
- Regular ethical audits and updates

## ⚠️ Limitations

- Cannot provide clinical diagnosis
- Not suitable for emergency situations
- Limited context understanding
- May not understand complex trauma
- Requires professional oversight
- Not a replacement for human therapists

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
