#!/bin/bash

s1='http://openconnecto.me/mrdata/share/dti/ndmg_v0011/MRN114/DS01216/MRN114_'
s2='_1_DTI_DS01216.graphml'

for i in 'M87165017' 'M87186642' 'M87105476' 'M87196591' 'M87192333' 'M87152844' 'M87141858' 'M87111924' 'M87150415' 'M87181216' 'M87154559'
do
	c=$s1$i$s2
	wget $c

done


