for fn in `ls *.onnx`; do
  ./MNNConvert -f ONNX --modelFile $fn --MNNModel ${fn%.*}.mnn --bizCode MNN
done