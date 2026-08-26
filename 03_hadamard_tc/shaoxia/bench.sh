
for VERSION in 3; do
    python bench.py \
        --rows 1024,2048,4096 \
        --dims 1024,2048,4096,8192 \
        --impl ${VERSION} \
        --name v${VERSION}
done

python compare.py --runs v1,v2,v3 --dtype all \
    --rows 1024,2048,4096 \
    --dims 1024,2048,4096,8192