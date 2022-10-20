apt-get update
apt-get install vim
cd /root/.cache
mkdir -p torch/hub/checkpoints
cd /workspace/users/zhaojing165/tools_pth
cp alexnet-owt-4df8aa71.pth /root/.cache/torch/hub/checkpoints/alexnet-owt-4df8aa71.pth
cp vgg16-397923af.pth /root/.cache/torch/hub/checkpoints/vgg16-397923af.pth
