#!/bin/bash
# Scraper for Yamaha e-Piano Competition dataset
[ ! "$1" ] && echo 'Error: please specify output dir' && exit
dir=$1
pages='https://www.piano-e-competition.com/midi_2002.asp
https://www.piano-e-competition.com/midi_2004.asp
https://www.piano-e-competition.com/midi_2006.asp
https://www.piano-e-competition.com/midi_2008.asp
https://www.piano-e-competition.com/midi_2009.asp
https://www.piano-e-competition.com/midi_20011.asp
'
mkdir -p $dir
for page in $pages; do
    for midi in $(curl -s $page | egrep -i '[^"]+\.mid' -o | sed 's/^\/*/\//g'); do
        echo "https://www.piano-e-competition.com$midi"
    done
done | wget -P $dir -i -
cd $dir
ls | egrep -v -i '\.mid$' | xargs rm
file * | grep -v 'Standard MIDI' | awk -F ':' '{print $1}' | xargs rm

