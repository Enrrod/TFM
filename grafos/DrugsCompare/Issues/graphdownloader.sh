#!/bin/bash

s1='http://openconnecto.me/mrdata/share/dti/ndmg_v0011/NKI1/DS01216/NKI24_'
s2='_1_DTI_DS01216.graphml'

for i in '1961098' '1793622' '4288245'
do
	c=$s1$i$s2
	wget $c
done


