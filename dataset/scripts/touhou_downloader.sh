#!/bin/bash
[ ! "$1" ] && echo 'Error: please specify output dir' && exit
dir=$1
for url in $(curl -s https://thwiki.cc/%E5%88%86%E7%B1%BB:%E5%AE%98%E6%96%B9MIDI \
    | egrep -o '[^"]+?\.mid' \
    | egrep '^/' \
    | sed 's/^/https:\/\/thwiki.cc/g' \
    | uniq);
do url=$(curl -s "$url" \
    | egrep -o '[^"]+?\.mid' \
    | egrep '^/' \
    | grep -v '%' \
    | sed 's/^/https:/g' \
    | uniq);
echo $url | tee /dev/stderr
done | uniq | wget -P $dir -i -
cd $dir
ls | egrep -i -v '\.mid$' | xargs rm
file * | grep -v 'Standard MIDI' | awk -F ':' '{print $1}' | xargs rm

