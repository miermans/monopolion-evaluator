
from monopolion_evaluator.cli import main


def test_main():
    main(['--training_data=tests/fixtures/toy_data_2player.gz', '--epochs=1'])


def test_main_validation():
    main([
        '--training_data=tests/fixtures/toy_data_2player.gz',
        '--validation_data=tests/fixtures/toy_data_2player.gz',
        '--epochs=1'])
