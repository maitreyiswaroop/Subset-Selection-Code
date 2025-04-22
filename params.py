PARAMS="--m1 4 \
--m 20 \
--dataset-size 1000 \
--noise-scale 0.1 \
--num-epochs 1000 \
--reg-type Quadratic_Barrier \
--reg-lambda 0.0352965711480333 \
--learning-rate 1.0 \
--batch-size 256 \
--optimizer-type sgd \
--seed 17 \
--patience 10 \
--alpha-init random_5 \
--estimator-type plugin \
--base-model-type rf \
--populations resnet resnet resnet \
--param-freezing"

# step scheduler
PARAMS="--m1 4 \
--m 20 \
--dataset-size 1000 \
--noise-scale 0.1 \
--num-epochs 1000 \
--reg-type Quadratic_Barrier \
--reg-lambda 0.0352965711480333 \
--learning-rate 1.0 \
--batch-size 256 \
--optimizer-type sgd \
--seed 17 \
--patience 10 \
--alpha-init random_5 \
--estimator-type plugin \
--base-model-type rf \
--populations resnet resnet resnet \
--param-freezing \
--scheduler step \
--scheduler-step-size 200 \
--scheduler-gamma 0.5"

# exp scheduler
PARAMS="--m1 4 \
--m 20 \
--dataset-size 1000 \
--noise-scale 0.1 \
--num-epochs 1000 \
--reg-type Quadratic_Barrier \
--reg-lambda 0.0352965711480333 \
--learning-rate 1.0 \
--batch-size 256 \
--optimizer-type sgd \
--seed 17 \
--patience 10 \
--alpha-init random_5 \
--estimator-type plugin \
--base-model-type rf \
--populations resnet resnet resnet \
--param-freezing \
--scheduler exponential \
--scheduler-gamma 0.997"

# cosine scheduler
PARAMS="--m1 4 \
--m 20 \
--dataset-size 1000 \
--noise-scale 0.1 \
--num-epochs 1000 \
--reg-type Quadratic_Barrier \
--reg-lambda 0.0352965711480333 \
--learning-rate 1.0 \
--batch-size 256 \
--optimizer-type sgd \
--seed 17 \
--patience 10 \
--alpha-init random_5 \
--estimator-type plugin \
--base-model-type rf \
--populations resnet resnet resnet \
--param-freezing \
--scheduler cosine \
--scheduler-min-lr 0.001"

# cyclic scheduler
PARAMS="--m1 4 \
--m 20 \
--dataset-size 1000 \
--noise-scale 0.1 \
--num-epochs 1000 \
--reg-type Quadratic_Barrier \
--reg-lambda 0.0352965711480333 \
--learning-rate 0.1 \
--batch-size 256 \
--optimizer-type sgd \
--seed 17 \
--patience 10 \
--alpha-init random_5 \
--estimator-type plugin \
--base-model-type rf \
--populations resnet resnet resnet \
--param-freezing \
--scheduler cyclic \
--base-lr 0.01 \
--max-lr 1.0 \
--step-size-up 100"

# one cycle
PARAMS="--m1 4 \
--m 20 \
--dataset-size 1000 \
--noise-scale 0.1 \
--num-epochs 1000 \
--reg-type Quadratic_Barrier \
--reg-lambda 0.0352965711480333 \
--learning-rate 1.0 \
--batch-size 256 \
--optimizer-type sgd \
--seed 17 \
--patience 10 \
--alpha-init random_5 \
--estimator-type plugin \
--base-model-type rf \
--populations resnet resnet resnet \
--param-freezing \
--scheduler one_cycle"

# plateau scheduler
PARAMS="--m1 4 \
--m 20 \
--dataset-size 1000 \
--noise-scale 0.1 \
--num-epochs 1000 \
--reg-type Quadratic_Barrier \
--reg-lambda 0.0352965711480333 \
--learning-rate 1.0 \
--batch-size 256 \
--optimizer-type sgd \
--seed 17 \
--patience 10 \
--alpha-init random_5 \
--estimator-type plugin \
--base-model-type rf \
--populations resnet resnet resnet \
--param-freezing \
--scheduler reduce_on_plateau \
--scheduler-patience 5 \
--scheduler-gamma 0.5 \
--scheduler-min-lr 0.0001"

# warmup + cosine scheduler
PARAMS="--m1 4 \
--m 20 \
--dataset-size 1000 \
--noise-scale 0.1 \
--num-epochs 1000 \
--reg-type Quadratic_Barrier \
--reg-lambda 0.0352965711480333 \
--learning-rate 1.0 \
--batch-size 256 \
--optimizer-type sgd \
--seed 17 \
--patience 10 \
--alpha-init random_5 \
--estimator-type plugin \
--base-model-type rf \
--populations resnet resnet resnet \
--param-freezing \
--scheduler warmup_cosine \
--warmup-epochs 100 \
--scheduler-min-lr 0.0001"