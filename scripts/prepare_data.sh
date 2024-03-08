#!/usr/bin/env bash

mkdir -p data
cd data
#gdown "https://drive.google.com/file/d/1L_We51j5FwIo5PeLvAK_9I3ZuFU_IWAq/view?usp=sharing"
gdown "https://drive.google.com/uc?id=1L_We51j5FwIo5PeLvAK_9I3ZuFU_IWAq"
unzip grabnet_data.zip
rm grabnet_data.zip
cd ..
mv data/grabnet_data/refinenet.pt grabnet/models/
mv data/grabnet_data/coarsenet.pt grabnet/models/
