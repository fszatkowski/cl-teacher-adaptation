from tests import run_main_and_assert

FAST_LOCAL_TEST_ARGS = "--exp-name local_test --datasets mnist" \
                       " --network LeNet --num-tasks 3 --seed 1 --batch-size 32" \
                       " --nepochs 3" \
                       " --num-workers 0" \
                       " --approach ssil"


def test_ssil():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num-exemplars 200"
    run_main_and_assert(args_line)


def test_ssil_ta():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num-exemplars 200 --ta"
    run_main_and_assert(args_line)
