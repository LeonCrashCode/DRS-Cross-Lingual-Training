
for lang in en de it nl
do
ln -s ../../pmb-4.0.0/exp_data/${lang}/gold/train.txt.raw.word.sent ${lang}.src
ln -s ../../pmb-4.0.0/exp_data/${lang}/gold/train.txt.word.tgt ${lang}.tgt 
ln -s ../../pmb-4.0.0/exp_data/${lang}/gold/train.txt ${lang}.txt
ln -s ../../pmb-4.0.0/exp_data/${lang}/gold/train.sbn ${lang}-sbn.tgt
ln -s ../../pmb-4.0.0/exp_data/${lang}/gold/train.txt.raw.word.sent-sbn ${lang}-sbn.src
ln -s ../../pmb-4.0.0/exp_data/${lang}/gold/train.penman ${lang}.penman

ln -s ../../pmb-4.0.0/exp_data/${lang}/silver/train.txt.raw.word.sent silver.${lang}.src
ln -s ../../pmb-4.0.0/exp_data/${lang}/silver/train.txt.word.tgt silver.${lang}.tgt
ln -s ../../pmb-4.0.0/exp_data/${lang}/silver/train.sbn silver.${lang}-sbn.tgt
ln -s ../../pmb-4.0.0/exp_data/${lang}/silver/train.txt.raw.word.sent-sbn silver.${lang}-sbn.src

done


cat silver.en.src silver.de.src silver.it.src silver.nl.src en.src de.src it.src nl.src > pretrain.src
cat silver.en.tgt silver.de.tgt silver.it.tgt silver.nl.tgt en.tgt de.tgt it.tgt nl.tgt > pretrain.tgt
cat silver.en-sbn.src silver.de-sbn.src silver.it-sbn.src silver.nl-sbn.src en-sbn.src de-sbn.src it-sbn.src nl-sbn.src > pretrain-sbn.src
cat silver.en-sbn.tgt silver.de-sbn.tgt silver.it-sbn.tgt silver.nl-sbn.tgt en-sbn.tgt de-sbn.tgt it-sbn.tgt nl-sbn.tgt > pretrain-sbn.tgt


for lang in en de it nl
do
ln -s ../../pmb-4.0.0/exp_data/${lang}/gold/dev.txt.raw.word.sent ${lang}.dev.src
ln -s ../../pmb-4.0.0/exp_data/${lang}/gold/dev.txt.word.tgt ${lang}.dev.tgt 
ln -s ../../pmb-4.0.0/exp_data/${lang}/gold/dev.txt ${lang}.dev.txt
ln -s ../../pmb-4.0.0/exp_data/${lang}/gold/dev.sbn ${lang}-sbn.dev.tgt
ln -s ../../pmb-4.0.0/exp_data/${lang}/gold/dev.txt.raw.word.sent-sbn ${lang}-sbn.dev.src
ln -s ../../pmb-4.0.0/exp_data/${lang}/gold/dev.penman ${lang}.dev.penman

done

cat en.dev.src de.dev.src it.dev.src nl.dev.src > dev.src
cat en.dev.tgt de.dev.tgt it.dev.tgt nl.dev.tgt > dev.tgt
cat en-sbn.dev.src de-sbn.dev.src it-sbn.dev.src nl-sbn.dev.src > dev-sbn.src
cat en-sbn.dev.tgt de-sbn.dev.tgt it-sbn.dev.tgt nl-sbn.dev.tgt > dev-sbn.tgt
