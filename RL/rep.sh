
p='-'
for n in `seq 10 2 30`; do
   for num in `seq 0.04 0.04 0.4`; do
    python ./q_progressive_recovery.py $n $num > ~/Desktop/result/"$n$p$num.txt"
  done
done

