
# this is a small derivative from 8M-big openwebtext dataset for testing

# how this build script and dataset_infos.json were generated

#

mkdir openwebtext-10k
cd openwebtext-10k

# data
wget https://zenodo.org/record/3834942/files/openwebtext.tar.xz
tar xf openwebtext.tar.xz
cd openwebtext
rename.pl 's|-|-00|; s|-00(\d\d\d)|-$1|; s|-00(\d\d)|-0$1|;' *xz

# now open the first 30 archives
mkdir subset
cp urlsf_subset00-0[0-2]*_data.xz subset
cd subset
find . -name "*xz" -exec tar xf {} \;
mkdir 10k
find . -name "*txt" | sort | head -10000 | xargs mv -t 10k
tar cfJ 10k.xz -C 10k .
mkdir openwebtext-10k
mv 10k.xz openwebtext-10k
tar cfJ openwebtext-10k.tar.xz openwebtext-10k
# the openwebtext subdir gets created on the fly
aws s3 cp openwebtext-10k.tar.xz s3://llm_datasets.huggingface.co/nlp/llm_datasets/openwebtext/

# script
wget https://raw.githubusercontent.com/huggingface/datasets/master/datasets/openwebtext/openwebtext.py
mv openwebtext.py openwebtext-10k.py
perl -pi -e 's|openwebtext|openwebtext-10k|g' openwebtext-10k.py
perl -pi -e 's|https://zenodo.org/record/3834942/files/|https://cdn-datasets.huggingface.co/nlp/datasets/openwebtext/|g' openwebtext-10k.py
perl -pi -e 's|Openwebtext|Openwebtext10k|g' openwebtext-10k.py



# manually check that the script is correct - edit the descriptions

# create a new dataset entry on the hub
https://huggingface.co/new-dataset

# once created clone it
git clone https://huggingface.co/datasets/stas/openwebtext-10k
cp openwebtext-10k.py process.txt openwebtext-10k
cd openwebtext-10k

git add openwebtext-10k.py process.txt
git commit -m "build script" openwebtext-10k.py process.txt
git push

# test and generate config file
cd ..
llm_datasets-cli test ./openwebtext-10k --save_infos --all_configs

# add and push the generated config
cd openwebtext-10k
git add dataset_infos.json
git commit -m "add dataset_infos.json" dataset_infos.json
git push

# test that the dataset is working
python -c "from datasets import load_dataset; ds=load_dataset('stas/openwebtext-10k'); print(ds)"