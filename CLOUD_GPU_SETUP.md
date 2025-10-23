# Cloud GPU Setup Guide - Quick Start

> Step-by-step instructions for setting up cloud GPU instances for CUDA learning. Essential for Mac users and anyone without local NVIDIA GPU.

---

## Quick Comparison

| Provider | Setup Time | Cost | Best For |
|----------|------------|------|----------|
| **Google Colab** | 2 minutes | FREE | Getting started |
| **Paperspace** | 10 minutes | $8-50/mo | Regular practice |
| **AWS EC2** | 20 minutes | Pay-as-you-go | Advanced projects |

---

## Option 1: Google Colab (Fastest Start)

### Step 1: Access Colab (2 minutes)

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Sign in with Google account
3. Click "New Notebook"

### Step 2: Enable GPU (30 seconds)

1. Click `Runtime` → `Change runtime type`
2. Select `GPU` from Hardware accelerator dropdown
3. Click `Save`

### Step 3: Test CUDA (1 minute)

In a code cell, run:
```python
# Check GPU availability
!nvidia-smi

# Check CUDA version
!nvcc --version
```

You should see output showing a Tesla T4 GPU!

### Step 4: Write and Run CUDA Code

```python
# Create a CUDA file
%%writefile hello.cu
#include <stdio.h>

__global__ void hello() {
    printf("Hello from GPU! Block %d, Thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {
    hello<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}

# Compile
!nvcc hello.cu -o hello

# Run
!./hello
```

### Step 5: Clone This Repo

```python
# Clone the learning repository
!git clone https://github.com/aimanyounises1/C_Plus_Plus_Advanced.git
%cd C_Plus_Plus_Advanced

# Compile and run examples
!nvcc solutions/phase5_cuda/01_vector_addition_optimized.cu -o vector_add -O3
!./vector_add
```

### Colab Limitations

⚠️ **Important:**
- Sessions timeout after 12 hours
- Can be disconnected if idle
- Files are deleted when session ends
- To save work: Download files or connect to Google Drive

### Persist Files to Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Work in Drive folder
%cd /content/drive/MyDrive/
!mkdir -p cuda_learning
%cd cuda_learning
```

---

## Option 2: Paperspace Gradient (Best for Serious Learning)

### Step 1: Sign Up (5 minutes)

1. Go to [paperspace.com/gradient](https://www.paperspace.com/gradient)
2. Create account (requires email + credit card)
3. Verify email

### Step 2: Create Notebook (3 minutes)

1. Click "Create" → "Notebook"
2. Choose runtime:
   - **Free tier:** P5000 (limited hours)
   - **Pro tier:** RTX 4000, A100, etc.
3. Select "PyTorch" or "TensorFlow" container (has CUDA pre-installed)
4. Click "Start Notebook"

### Step 3: Access Terminal (1 minute)

Once notebook starts:
1. Click "File" → "New" → "Terminal"
2. You now have full bash access with CUDA!

### Step 4: Setup Environment (2 minutes)

```bash
# Check CUDA
nvidia-smi
nvcc --version

# Install tools
apt-get update
apt-get install -y git cmake

# Clone repo
git clone https://github.com/aimanyounises1/C_Plus_Plus_Advanced.git
cd C_Plus_Plus_Advanced
```

### Step 5: Compile and Run

```bash
cd solutions/phase5_cuda

# Compile with appropriate architecture
# P5000 = sm_61, RTX 4000 = sm_75, A100 = sm_80
nvcc -o vector_add 01_vector_addition_optimized.cu -O3 -arch=sm_75

# Run
./vector_add
```

### Step 6: Setup SSH (Optional but Recommended)

1. Go to "Settings" in Paperspace
2. Add your SSH public key
3. Connect from terminal:
```bash
ssh paperspace@<your-instance-id>.paperspacegradient.com
```

### Paperspace Pro Tips

✅ **Enable Auto-shutdown:**
- Settings → Set auto-shutdown to 1 hour
- Saves money when you forget to stop

✅ **Use Persistent Storage:**
- Files in `/storage` persist between sessions
- Put your projects here

```bash
mkdir -p /storage/cuda_learning
cd /storage/cuda_learning
git clone https://github.com/aimanyounises1/C_Plus_Plus_Advanced.git
```

---

## Option 3: AWS EC2 (Most Powerful)

### Prerequisites (5 minutes)

1. AWS account with payment method
2. Basic understanding of cloud concepts

### Step 1: Launch Instance (10 minutes)

1. Go to [AWS Console](https://console.aws.amazon.com)
2. Navigate to EC2 Dashboard
3. Click "Launch Instance"

**Configuration:**
- **Name:** `cuda-learning`
- **AMI:** Search for "Deep Learning AMI (Ubuntu 20.04)" (has CUDA pre-installed)
- **Instance type:** `g4dn.xlarge` (cheapest GPU option, ~$0.50/hour)
- **Key pair:** Create new or select existing
- **Network:** Allow SSH (port 22)

4. Click "Launch Instance"

### Step 2: Connect via SSH (2 minutes)

```bash
# Make key readable only by you
chmod 400 your-key.pem

# Connect
ssh -i your-key.pem ubuntu@<instance-public-ip>
```

Find public IP in EC2 console under "Instances".

### Step 3: Verify CUDA (1 minute)

```bash
# Should work immediately with Deep Learning AMI
nvidia-smi
nvcc --version

# Test compilation
echo '#include <stdio.h>
__global__ void test() { printf("GPU works!\n"); }
int main() { test<<<1,1>>>(); cudaDeviceSynchronize(); return 0; }
' > test.cu

nvcc test.cu -o test
./test
```

### Step 4: Setup Development Environment (5 minutes)

```bash
# Update system
sudo apt update

# Install tools
sudo apt install -y git cmake tmux htop

# Clone repo
git clone https://github.com/aimanyounises1/C_Plus_Plus_Advanced.git
cd C_Plus_Plus_Advanced

# Compile examples
cd solutions/phase5_cuda
nvcc -o vector_add 01_vector_addition_optimized.cu -O3 -arch=sm_75
./vector_add
```

### Step 5: Setup VS Code Remote (Optional, 10 minutes)

On your local machine:

1. Install VS Code and "Remote - SSH" extension
2. Add to `~/.ssh/config`:
```
Host aws-gpu
    HostName <your-instance-ip>
    User ubuntu
    IdentityFile ~/.ssh/your-key.pem
```
3. In VS Code: Connect to `aws-gpu`
4. Develop as if local!

### Step 6: Important - Stop Instance When Done!

```bash
# From local machine
aws ec2 stop-instances --instance-ids i-your-instance-id

# Or use AWS Console:
# EC2 → Instances → Right-click → Instance State → Stop
```

⚠️ **Critical:** Stopped instances cost almost nothing. Running instances cost $0.50/hour = $360/month!

### AWS Cost Optimization

**Use Spot Instances (70% cheaper):**
- Same as above but select "Spot Instance" option
- Can be interrupted (fine for learning)
- ~$0.15/hour instead of $0.50/hour

**Set Billing Alerts:**
1. AWS Console → Billing → Budgets
2. Create budget alert (e.g., $50/month)
3. Get email when approaching limit

---

## Option 4: Google Cloud Platform (Alternative to AWS)

### Quick Setup

1. Go to [cloud.google.com](https://cloud.google.com)
2. Create account ($300 free credits!)
3. Enable Compute Engine API
4. Create VM with GPU:

```bash
gcloud compute instances create cuda-vm \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=common-cu113 \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE \
    --boot-disk-size=100GB
```

5. SSH:
```bash
gcloud compute ssh cuda-vm
```

---

## Comparison: Which Cloud to Choose?

### For Beginners:
**Google Colab FREE**
- ✅ Instant access
- ✅ No cost
- ✅ Good for learning
- ❌ Limited sessions

### For Regular Practice:
**Paperspace Gradient**
- ✅ Persistent storage
- ✅ Easy to use
- ✅ Good GPUs
- ✅ Fair pricing
- Recommended!

### For Serious Projects:
**AWS EC2**
- ✅ Best GPU selection
- ✅ Full control
- ✅ Industry-standard
- ❌ More complex
- ❌ Easy to forget and rack up costs

### For Free Credits:
**GCP or AWS**
- ✅ $300 free credits (new accounts)
- ✅ Can last 3-6 months of learning
- Use for major projects

---

## Development Workflow Examples

### Workflow 1: Colab for Quick Tests

```python
# In Colab notebook
%%writefile kernel.cu
// Your CUDA code

!nvcc kernel.cu -o kernel -O3
!./kernel
```

**Use for:**
- Testing concepts
- Quick experiments
- Sharing with others (share Colab link)

### Workflow 2: Paperspace + VS Code Remote

```bash
# On local Mac:
code --remote ssh-remote+paperspace /storage/cuda_learning

# Edit files locally, run on GPU remotely!
```

**Use for:**
- Daily development
- Longer projects
- Comfortable coding experience

### Workflow 3: AWS + tmux for Long Jobs

```bash
# SSH to AWS
ssh aws-gpu

# Start tmux session
tmux new -s training

# Run long job
./long_training_job

# Detach: Ctrl-B then D
# Job keeps running!

# Reconnect later
tmux attach -s training
```

**Use for:**
- Training runs
- Benchmarking
- Project development

---

## Troubleshooting

### "CUDA out of memory"
```bash
# Check GPU memory
nvidia-smi

# Free memory by killing processes
nvidia-smi | grep python
kill <PID>
```

### "nvcc: command not found"
```bash
# Add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Add to ~/.bashrc to persist
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
```

### "Driver version insufficient"
The cloud AMI should have correct drivers. If not:
```bash
# Check driver
nvidia-smi

# Update if needed (rarely necessary on cloud)
sudo apt install nvidia-driver-525
```

### SSH Connection Issues
```bash
# Check security group allows port 22
# Check key permissions: chmod 400 key.pem
# Check username: usually 'ubuntu' or 'ec2-user'
```

---

## Cost Calculator

### Monthly Cost Estimates (Assuming 2 hours/day practice)

| Provider | Setup | Daily Use | Monthly Total |
|----------|-------|-----------|---------------|
| Colab Free | $0 | $0 | $0 |
| Colab Pro | $10 | $0 | $10 |
| Paperspace (P5000) | $8 | $0.51/hr × 2hr × 30 | $39 |
| AWS g4dn.xlarge | $0 | $0.50/hr × 2hr × 30 | $30 |
| AWS spot (70% off) | $0 | $0.15/hr × 2hr × 30 | $9 |

**Recommendation:** Start free, upgrade as needed.

---

## Best Practices

### 1. Always Set Auto-Shutdown
Prevent accidentally leaving instances running.

### 2. Use Version Control
```bash
git add .
git commit -m "Progress update"
git push
```
Never lose work even if instance terminates.

### 3. Monitor Costs
- Check daily
- Set billing alerts
- Stop when not using

### 4. Backup Important Work
```bash
# Download from cloud
scp -r aws-gpu:~/project ./local-backup

# Or use rsync
rsync -avz aws-gpu:~/project ./local-backup
```

### 5. Use tmux/screen
Never lose work due to connection drops.

---

## Quick Command Reference

### Colab
```python
# Check GPU
!nvidia-smi

# Compile CUDA
!nvcc file.cu -o program -O3

# Run
!./program
```

### Paperspace/AWS/GCP
```bash
# Check GPU
nvidia-smi

# Compile
nvcc file.cu -o program -O3 -arch=sm_75

# Profile
ncu ./program
nsys profile ./program

# Stop (save money!)
# AWS: aws ec2 stop-instances --instance-ids i-xxx
# GCP: gcloud compute instances stop instance-name
```

---

## Getting Help

**GPU not showing:**
- Check runtime type in Colab
- Check instance type in cloud (must include GPU)

**Compilation errors:**
- Check CUDA version: `nvcc --version`
- Check GPU arch: `nvidia-smi` then use appropriate `-arch=sm_XX`

**Cloud costs too high:**
- Use Colab Free for learning
- Stop instances when not using
- Use spot/preemptible instances

---

## Next Steps

1. **Choose your platform** (Colab to start)
2. **Complete setup** (5-20 minutes depending on platform)
3. **Clone this repo**
4. **Start with** `solutions/phase5_cuda/01_vector_addition_optimized.cu`
5. **Practice daily** (2 hours recommended)

---

## Resources

- [Google Colab GPU FAQ](https://research.google.com/colaboratory/faq.html)
- [Paperspace Documentation](https://docs.paperspace.com)
- [AWS EC2 GPU Instances](https://aws.amazon.com/ec2/instance-types/p3/)
- [GCP GPU Pricing](https://cloud.google.com/compute/gpus-pricing)

---

**Remember:** The goal is to learn CUDA, not to become a cloud expert. Start with Colab, upgrade when you need more power. The knowledge you gain is what matters for Nvidia interviews! 🚀
