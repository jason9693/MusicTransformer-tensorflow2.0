#!/bin/bash

[ ! "$1" ] && echo 'Error: please specify output dir' && exit
[ ! "$2" ] && echo 'Error: please specify page url' && exit

echo "$(curl -s $2 | egrep -o 'http.+download/[^"]+' | uniq)" \
    | wget --content-disposition -P $1 -i -

cd $1
ls | egrep -i -v '\.mid$' | xargs rm
file * | grep -v 'Standard MIDI' | awk -F ':' '{print $1}' | xargs rm
