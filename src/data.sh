
lang=${1}
for i in train dev test
do
ln -s ../../pmb-4.0.0/exp_data/${lang}/gold/${i}.txt.raw.word.sent ${i}.src
ln -s ../../pmb-4.0.0/exp_data/${lang}/gold/${i}.txt.word.tgt ${i}.tgt 
ln -s ../../pmb-4.0.0/exp_data/${lang}/gold/${i}.txt
ln -s ../../pmb-4.0.0/exp_data/${lang}/gold/${i}.sbn ${i}-sbn.tgt
ln -s ../../pmb-4.0.0/exp_data/${lang}/gold/${i}.sbn_simple ${i}-sbn_simple.tgt
ln -s ${i}-sbn.src ${i}-sbn_simple.src
ln -s ../../pmb-4.0.0/exp_data/${lang}/gold/${i}.txt.raw.word.sent-sbn ${i}-sbn.src
ln -s ../../pmb-4.0.0/exp_data/${lang}/gold/${i}.penman
done

ln -s ../../pmb-4.0.0/exp_data/${lang}/silver/train.txt.raw.word.sent silver.src
ln -s ../../pmb-4.0.0/exp_data/${lang}/silver/train.txt.word.tgt silver.tgt
ln -s ../../pmb-4.0.0/exp_data/${lang}/silver/train.sbn silver-sbn.tgt
ln -s ../../pmb-4.0.0/exp_data/${lang}/silver/train.sbn_simple silver-sbn_simple.tgt
ln -s ../../pmb-4.0.0/exp_data/${lang}/silver/train.txt.raw.word.sent-sbn silver-sbn.src
ln -s silver-sbn.src silver-sbn_simple.src

cat silver.src train.src > pretrain.src
cat silver.tgt train.tgt > pretrain.tgt
cat silver-sbn.tgt train-sbn.tgt > pretrain-sbn.tgt
cat silver-sbn.src train-sbn.src > pretrain-sbn.src
cat silver-sbn_simple.tgt train-sbn_simple.tgt > pretrain-sbn_simple.tgt
ln -s pretrain-sbn.src pretrain-sbn_simple.src
