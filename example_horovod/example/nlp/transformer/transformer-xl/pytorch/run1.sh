scp /home/user/eurosys23/workspace/ACTopk/examples/nlp/transformer/transformer-xl/pytorch/train.py user@n16:/home/user/eurosys23/workspace/ACTopk/examples/nlp/transformer/transformer-xl/pytorch/
scp /home/user/eurosys23/workspace/ACTopk/examples/nlp/transformer/transformer-xl/pytorch/train.py user@n17:/home/user/eurosys23/workspace/ACTopk/examples/nlp/transformer/transformer-xl/pytorch/
scp /home/user/eurosys23/workspace/ACTopk/examples/nlp/transformer/transformer-xl/pytorch/train.py user@n18:/home/user/eurosys23/workspace/ACTopk/examples/nlp/transformer/transformer-xl/pytorch/
scp /home/user/eurosys23/workspace/ACTopk/examples/nlp/transformer/transformer-xl/pytorch/train.py user@n19:/home/user/eurosys23/workspace/ACTopk/examples/nlp/transformer/transformer-xl/pytorch/
scp /home/user/eurosys23/workspace/ACTopk/examples/nlp/transformer/transformer-xl/pytorch/train.py user@n20:/home/user/eurosys23/workspace/ACTopk/examples/nlp/transformer/transformer-xl/pytorch/
scp /home/user/eurosys23/workspace/ACTopk/examples/nlp/transformer/transformer-xl/pytorch/train.py user@n21:/home/user/eurosys23/workspace/ACTopk/examples/nlp/transformer/transformer-xl/pytorch/
scp /home/user/eurosys23/workspace/ACTopk/examples/nlp/transformer/transformer-xl/pytorch/train.py user@n22:/home/user/eurosys23/workspace/ACTopk/examples/nlp/transformer/transformer-xl/pytorch/

scp /home/user/eurosys23/workspace/ACTopk/examples/nlp/transformer/transformer-xl/pytorch/train_gtopk.py user@n16:/home/user/eurosys23/workspace/ACTopk/examples/nlp/transformer/transformer-xl/pytorch/
scp /home/user/eurosys23/workspace/ACTopk/examples/nlp/transformer/transformer-xl/pytorch/train_gtopk.py user@n17:/home/user/eurosys23/workspace/ACTopk/examples/nlp/transformer/transformer-xl/pytorch/
scp /home/user/eurosys23/workspace/ACTopk/examples/nlp/transformer/transformer-xl/pytorch/train_gtopk.py user@n18:/home/user/eurosys23/workspace/ACTopk/examples/nlp/transformer/transformer-xl/pytorch/
scp /home/user/eurosys23/workspace/ACTopk/examples/nlp/transformer/transformer-xl/pytorch/train_gtopk.py user@n19:/home/user/eurosys23/workspace/ACTopk/examples/nlp/transformer/transformer-xl/pytorch/
scp /home/user/eurosys23/workspace/ACTopk/examples/nlp/transformer/transformer-xl/pytorch/train_gtopk.py user@n20:/home/user/eurosys23/workspace/ACTopk/examples/nlp/transformer/transformer-xl/pytorch/
scp /home/user/eurosys23/workspace/ACTopk/examples/nlp/transformer/transformer-xl/pytorch/train_gtopk.py user@n21:/home/user/eurosys23/workspace/ACTopk/examples/nlp/transformer/transformer-xl/pytorch/
scp /home/user/eurosys23/workspace/ACTopk/examples/nlp/transformer/transformer-xl/pytorch/train_gtopk.py user@n22:/home/user/eurosys23/workspace/ACTopk/examples/nlp/transformer/transformer-xl/pytorch/

scp /home/user/eurosys23/workspace/ACTopk/examples/nlp/transformer/transformer-xl/pytorch/train_actopk.py user@n16:/home/user/eurosys23/workspace/ACTopk/examples/nlp/transformer/transformer-xl/pytorch/
scp /home/user/eurosys23/workspace/ACTopk/examples/nlp/transformer/transformer-xl/pytorch/train_actopk.py user@n17:/home/user/eurosys23/workspace/ACTopk/examples/nlp/transformer/transformer-xl/pytorch/
scp /home/user/eurosys23/workspace/ACTopk/examples/nlp/transformer/transformer-xl/pytorch/train_actopk.py user@n18:/home/user/eurosys23/workspace/ACTopk/examples/nlp/transformer/transformer-xl/pytorch/
scp /home/user/eurosys23/workspace/ACTopk/examples/nlp/transformer/transformer-xl/pytorch/train_actopk.py user@n19:/home/user/eurosys23/workspace/ACTopk/examples/nlp/transformer/transformer-xl/pytorch/
scp /home/user/eurosys23/workspace/ACTopk/examples/nlp/transformer/transformer-xl/pytorch/train_actopk.py user@n20:/home/user/eurosys23/workspace/ACTopk/examples/nlp/transformer/transformer-xl/pytorch/
scp /home/user/eurosys23/workspace/ACTopk/examples/nlp/transformer/transformer-xl/pytorch/train_actopk.py user@n21:/home/user/eurosys23/workspace/ACTopk/examples/nlp/transformer/transformer-xl/pytorch/
scp /home/user/eurosys23/workspace/ACTopk/examples/nlp/transformer/transformer-xl/pytorch/train_actopk.py user@n22:/home/user/eurosys23/workspace/ACTopk/examples/nlp/transformer/transformer-xl/pytorch/


bash run_hvd_w103.sh train ; 
bash run_hvd_w103_gtopk.sh train ; 
bash run_hvd_w103_actopk.sh train ;