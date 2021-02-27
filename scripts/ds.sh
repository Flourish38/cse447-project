src=$1
dest=$2

mkdir $2
for i in $src/*.txt; do
  out="$dest"/$(basename $i)
  lines=`wc -l $i | cut -d " " -f1`
  if expr lines > 30000; then
    lines=$(expr $lines / 20)
  fi
  cat $i | shuf | head -n $(expr $lines) > $out
  echo $i
  echo $out
done