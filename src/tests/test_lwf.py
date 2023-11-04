from tests import run_main_and_assert

FAST_LOCAL_TEST_ARGS = "--exp-name local_test --datasets mnist" \
                       " --network LeNet --num-tasks 3 --seed 1 --batch-size 32" \
                       " --nepochs 3" \
                       " --num-workers 0" \
                       " --approach lwf"


def test_lwf_without_exemplars():
    run_main_and_assert(FAST_LOCAL_TEST_ARGS)


def test_lwf_with_exemplars():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num-exemplars 200"
    run_main_and_assert(args_line)


def test_lwf_with_warmup():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --warmup-nepochs 5"
    args_line += " --warmup-lr-factor 0.5"
    args_line += " --num-exemplars 200"
    run_main_and_assert(args_line)


def test_lwf_ta():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --ta"
    run_main_and_assert(args_line)


def test_lwf_warmup():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --wu-nepochs 1"
    run_main_and_assert(args_line)


def test_lwf_warmup_with_patience():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --wu-nepochs 2 --wu-patience 1"
    run_main_and_assert(args_line)


def test_lwf_warmup_cosine():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --wu-nepochs 2 --wu-scheduler cosine"
    run_main_and_assert(args_line)


def test_lwf_warmup_onecycle():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --wu-nepochs 2 --wu-scheduler onecycle"
    run_main_and_assert(args_line)


def test_lwf_warmup_plateau():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --wu-nepochs 2 --wu-scheduler plateau"
    run_main_and_assert(args_line)


def test_lwf_with_distillation_from_warmup_head():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --wu-nepochs 1 --distill-from-warmup-head"
    run_main_and_assert(args_line)


def test_lwf_mc():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --mc"
    run_main_and_assert(args_line)
