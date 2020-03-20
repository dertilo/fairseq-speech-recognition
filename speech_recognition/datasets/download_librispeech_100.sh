%%bash
base_url=www.openslr.org/resources/12
train_dir=train_100

download_dir=${1%/}
out_dir=${2%/}

#download_dir=/content/librispeech_raw
#out_dir=/content/fairseq_preprocessed_librispeech

mkdir -p ${out_dir}
cd ${out_dir} || exit

echo "Data Download"
for part in dev-clean test-clean dev-other test-other train-clean-100; do
    url=$base_url/$part.tar.gz
    if ! wget -P $download_dir $url; then
        echo "$0: wget failed for $url"
        exit 1
    fi
    if ! tar -C $download_dir -xzf $download_dir/$part.tar.gz; then
        echo "$0: error un-tarring archive $download_dir/$part.tar.gz"
        exit 1
    fi
done