#!/bin/bash

# this script creates a collage of observation gifs

for fstate in *.state.gif; do
  fbase=${fstate%.*.*}
  echo "processing $fbase"

  # split observation gifs
  for fobservation in $fbase.observation.{full,partial,minigrid,raytracing}.gif; do
    echo "splitting $fobservation"
    convert -coalesce $fobservation ${fobservation%.*}.png
  done

  # add caption text to pngs
  for fobservation in $fbase.observation.*.png; do
    echo "annotating $fobservation"
    text=${fobservation%-*}
    text=${text##*.}
    text="${text^} Observation"
    convert -pointsize 20 -undercolor black -fill yellow -draw "text 10,270 '$text'" $fobservation $fobservation
  done

  # make montage of annotated pngs
  nframes=$(identify $fbase.state.gif | wc -l)
  for frame in $(seq 0 $(($nframes - 1))); do
    echo "combining $fbase.observation.\{full,partial,minigrid,raytracing\}-$frame.png"
    montage $fbase.observation.{full,partial,minigrid,raytracing}-$frame.png -background "#888" -geometry +5+5 $fbase.observation.montage-$frame.png
    rm $fbase.observation.{full,partial,minigrid,raytracing}-$frame.png
  done

  # reconstruct montage gif
  frames=$(find . -name "$fbase.observation.montage-*.png"  | sort -V)
  echo "making $fbase.observation.montage.gif"
  convert -delay 100 -loop 0 $frames $fbase.observation.montage.gif
  rm $frames

done
