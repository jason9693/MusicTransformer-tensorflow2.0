#!/bin/bash
# Scraper for Classical Piano Midi Page
[ ! "$1" ] && echo 'Error: please specify output dir' && exit
dir=$1
base=http://www.piano-midi.de
pages=$(curl -s --max-time 5 $base/midi_files.htm \
    | grep '<tr class="midi"><td class="midi"><a href="' \
    | egrep '[^"]+\.htm' -o)
echo Pages: $pages
mkdir -p $dir
for page in $pages; do
    midis=$(curl -s --max-time 5 $base/$page | egrep '[^"]+format0\.mid' -o)
    for midi in $midis; do
        echo "http://www.piano-midi.de/$midi"
    done | tee /dev/stderr | wget -P $dir -i -
done
cd $dir
ls | egrep -v -i '\.mid$' | xargs rm
file * | grep -v 'Standard MIDI' | awk -F ':' '{print $1}' | xargs rm

