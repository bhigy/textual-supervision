for exp in runs/asr*-ds*; do
    cd $exp
    python ../../evaluate_net.py -b net.best.pt >result_beam.json
    cd - > /dev/null
done
