#!/bin/bash

s1='http://openconnecto.me/mrdata/share/dti/ndmg_v0011/NKI1/DS01216/NKI24_'
s2='_1_DTI_DS01216.graphml'

for i in '2475376' '9630905' '2842950' '2799329' '8735778' '7055197' '3201815' '0021006' '0021024' '0021001' '0021018' '3795193' '4176156' '3315657' '0021002'
do
	c=$s1$i$s2
	wget $c
done
