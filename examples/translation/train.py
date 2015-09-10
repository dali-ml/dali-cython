import argparse
import dali.core as D
from dali.data import Lines, Process, DiscoverFiles, BatchBenefactor, IdentityReducer
from dali.data.batch import TranslationBatch
from dali.data.translation import TranslationFiles, TranslationMapper, build_vocabs, iterate_examples
from dali.utils.scoring import bleu, ErrorTracker
from dali.utils import (
    Vocab,
    Solver,
    median_smoothing,
    subsample,
    Throttled,
    pickle_from_scope,
    unpickle_as_dict,
    set_device_from_args,
    add_device_args,
)
import math
import os
import sys
import time
import random


from translation import TranslationModel

def parse_args():
    parser = argparse.ArgumentParser()
    # device
    add_device_args(parser)

    # paths and data
    parser.add_argument("--train",     type=str, required=True)
    parser.add_argument("--validate",  type=str, required=True)
    parser.add_argument("--save",      type=str, default=None)
    parser.add_argument("--from_lang", type=str, required=True)
    parser.add_argument("--to_lang",   type=str, required=True)
    parser.add_argument("--max_from_vocab", type=int, default=20000)
    parser.add_argument("--max_to_vocab",   type=int, default=20000)

    # training
    parser.add_argument("--minibatch",           type=int, default=64)
    parser.add_argument("--max_sentence_length", type=int, default=40)

    # model
    parser.add_argument("--input_size",   type=int, default=512)
    parser.add_argument("--hidden_sizes", type=int, nargs='+', default=[512,512,512,512])

    # solver
    parser.add_argument("--solver_type",   type=str, default="sgd")
    parser.add_argument("--learning_rate", type=int, default=0.003)

    return parser.parse_args()

def load_raw_validation(path, from_lang, to_lang, max_sentence_length):
    """List of validation sentences as strings.

    Used for reconstructions and BLEU.
    """
    p = Process(files=TranslationFiles(path, from_lang, to_lang),
            mapper=TranslationMapper(sentence_bounds=(0, max_sentence_length)),
            reducer=IdentityReducer())
    return list(p)

def show_reconstructions(model, example_pair, vocabs, max_sentence_length):
    from_words, to_words = example_pair
    from_vocab, to_vocab = vocabs
    from_with_unk = ' '.join(from_vocab.decode(from_vocab.encode(from_words)))
    to_with_unk   = ' '.join(to_vocab.decode(to_vocab.encode(to_words)))
    print('TRANSLATING: %s' % from_with_unk)
    print('REFERENCE:   %s' % to_with_unk)
    print('')
    for solution, score, _ in model.predict(from_vocab.encode(list(reversed(from_words)), add_eos=False),
                                           eos_symbol=to_vocab.eos,
                                           max_sequence_length=max_sentence_length + 1):
        score = math.exp(score.w[0])
        # reveal the unks
        solution = ' '.join(to_vocab.decode(solution, False))
        print('    %f => %s' % (score, to_vocab.decode(solution, True)))

def main(args):
    set_device_from_args(args, verbose=True)

    ############### MODEL/DATA LOADING ####################

    RELEVANT_VARIABLES = ["model", "vocabs", "solver", "data", "train_error", "validate_error"]

    if args.save is not None and os.path.exists(args.save):
        print("Resuming saved experiment at %s." % (args.save,))
        loaded = unpickle_as_dict(args.save)
        model, vocabs, solver, data, train_error, validate_error = [loaded[x] for x in RELEVANT_VARIABLES]
        solver.parameters = model.parameters()
    else:
        print("Loading vocabs - for monstrous datasets hit ctrl+C after you feel like probably enough words have been sampled.")
        vocabs = build_vocabs(args.train, args.from_lang, args.to_lang,
                              from_max_size=args.max_from_vocab,
                              to_max_size=args.max_to_vocab)
        print("Creating model")
        model  = TranslationModel(args.input_size,
                                  args.hidden_sizes,
                                  len(vocabs[0]),
                                  len(vocabs[1]))
        model.name_parameters("model")
        solver = Solver(model.parameters(), "sgd", learning_rate=0.003)

        solver.set_lr_multiplier("model.encoder_embedding", 2)
        solver.set_lr_multiplier("model.decoder_embedding", 2)

        data             = []
        train_error      = ErrorTracker()
        validate_error   = ErrorTracker()

    from_vocab, to_vocab = vocabs

    print("Input size:   ",        args.input_size)
    print("Hidden sizes: ",        args.hidden_sizes)
    print("max sentence length: ", args.max_sentence_length)

    print (args.from_lang + " vocabulary containts", len(from_vocab), "words")
    print (args.to_lang   + " vocabulary containts", len(to_vocab),   "words")


    def create_dataset_iterator(dataset, sentences_until_minibatch):
        return iterate_examples(dataset, args.from_lang, args.to_lang, vocabs,
                                minibatch_size=args.minibatch,
                                sentence_length_bounds=(0, args.max_sentence_length),
                                sentences_until_minibatch=sentences_until_minibatch)


    validation_pairs_text = load_raw_validation(args.validate, args.from_lang, args.to_lang, args.max_sentence_length)
    validation_batches    = list(create_dataset_iterator(args.validate, args.minibatch))


    t = Throttled(10)

    while True:
        total_time  = 0.0
        num_words, num_batches = 0, 0

        if solver.solver_type == 'adagrad':
            solver.reset_caches(params)

        for batch in data:
            batch_start_time = time.time()
            error = model.error(batch)
            (error / batch.examples).grad()
            D.Graph.backward()

            solver.step()
            batch_end_time = time.time()

            train_error.append(error / batch.to_tokens)

            total_time += batch_end_time - batch_start_time
            num_words   += batch.from_tokens + batch.to_tokens
            num_batches += 1

            if num_batches % 10 == 0:
                val_batch = random.choice(validation_batches)
                with D.NoBackprop():
                    validate_error.append(model.error(val_batch) / val_batch.to_tokens)

            if t.should_i_run() and num_batches > 0 and abs(total_time) > 1e-6:
                print('Epochs completed:  ', train_error.num_epochs())
                print('Error:             ', train_error.recent(10))
                print('Time per batch:    ', total_time  / num_batches)
                print('Words per second:  ', num_words   / total_time )
                print('Batches processed: ', num_batches)
                if hasattr(solver, 'step_size'):
                    print('Solver step size:  ', solver.step_size)
                show_reconstructions(model, random.choice(validation_pairs_text), vocabs, args.max_sentence_length)
                sys.stdout.flush()

            # free memory as soon as possible
            del batch

        train_error.finalize_epoch()
        validate_error.finalize_epoch()
        if train_error.num_epochs() > 0 and args.save is not None:
            print("Saving model to %s." % (args.save,))
            pickle_from_scope(args.save, RELEVANT_VARIABLES)

        data = create_dataset_iterator(args.train, 1000 * args.minibatch)

if __name__ == '__main__':
    main(parse_args())
