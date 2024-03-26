#!/bin/bash
###  Job name
#PBS -N example
#PBS -l nodes=1:ppn=16
#PBS -q q_6666
#PBS -o $PBS_JOBID
#PBS -r n

cd $PBS_O_WORKDIR

export OMP_NUM_THREADS=4
source /home/client5/workdir/wh/chem/bin/activate
source /public/home/client10/g16.sh

# Record time
echo " start at " $(date) > $PBS_O_WORKDIR/time

mkdir xyz
mkdir xyz/xyz_md

# Generate initial structure
python Smiles2Struct.py --type 0

# Submit MD job using GFN0
cd xyz
for file in *.xyz
do
        mkdir ${file%.*}
        cp $file ../md_param.inp ${file%.*}
	cd ${file%.*}
	xtb $file --input md_param.inp --omd --gfn 0 --namespace ./${file%.*} > xtb_out #strip file's extension
	cp *-*.xtb.trj ../xyz_md
	rm *.charges *.mdrestart *.wbo *.xtbmdok *.xtbopt.log *scoo* *mol *xtbrestart
	rm ../*xtboptok

# Submit MD job using GFN2
	cd ../xyz_md
	mkdir ${file%.*}
	cp ${file%.*}.xtb.trj ${file%.*}
	cd ${file%.*}
	crest -mdopt ${file%.*}.xtb.trj -gfn0 -opt normal  > crest_out
	echo -e '0.5\n0.2\n' | /home/client5/workdir/wh/software/molclus_1.9.9.9_Linux/isostat crest_ensemble.xyz
	crest -mdopt cluster.xyz -gfn2 -opt normal  >> crest_out
	echo -e "\n" | /home/client5/workdir/wh/software/molclus_1.9.9.9_Linux/isostat crest_ensemble.xyz -Nout 1 -Edis 0.5 -Gdis 0.2
	rm coor* stru*
	mv cluster.xyz $file
	
# Submit optimize job using Gaussian
	cat > ${file%.*}.com << EOF
%chk=${file%.*}.chk
%mem=16GB
%nprocshared=16
# opt freq b3lyp/6-31g** em=gd3bj 

${file%.*}

0 1
EOF
	tail -n +3 $file >> ${file%.*}.com
	dos2unix ${file%.*}.com
	cat >> ${file%.*}.com << EOF

EOF
	g16 ${file%.*}.com 
	formchk ${file%.*}.chk
	cd ../../..
done

echo "end at " $(date) >> $PBS_O_WORKDIR/time
