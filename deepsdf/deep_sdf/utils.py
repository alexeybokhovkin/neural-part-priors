#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import logging
import torch


def add_common_args(arg_parser):
    arg_parser.add_argument(
        "--debug",
        dest="debug",
        default=False,
        action="store_true",
        help="If set, debugging messages will be printed",
    )
    arg_parser.add_argument(
        "--quiet",
        "-q",
        dest="quiet",
        default=False,
        action="store_true",
        help="If set, only warnings will be printed",
    )
    arg_parser.add_argument(
        "--log",
        dest="logfile",
        default=None,
        help="If set, the log will be saved using the specified filename.",
    )


def configure_logging(args):
    logger = logging.getLogger()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)
    logger_handler = logging.StreamHandler()
    formatter = logging.Formatter("DeepSdf - %(levelname)s - %(message)s")
    logger_handler.setFormatter(formatter)
    logger.addHandler(logger_handler)

    if args.logfile is not None:
        file_logger_handler = logging.FileHandler(args.logfile)
        file_logger_handler.setFormatter(formatter)
        logger.addHandler(file_logger_handler)


def decode_sdf(decoder, latent_vector, queries, add_feat=None, mode=0):
    num_samples = queries.shape[0]

    if mode == 2:
        latent_repeat = latent_vector.expand(num_samples, -1)
        inputs = torch.cat([latent_repeat, queries], 1)
    elif mode == 1:
        add_feat = add_feat.expand(num_samples, -1)
        latent_repeat = latent_vector.expand(num_samples, -1)
        inputs = torch.cat([add_feat, latent_repeat, queries], 1).cuda()
    elif mode == 3:
        add_feat = add_feat.expand(num_samples, -1)
        inputs = torch.cat([add_feat, queries], 1).cuda()

    sdf = decoder(inputs)

    return sdf
