

def adda_args(parser):
    adda_parser = parser.add_argument_group("ADDA options")

    adda_parser.add_argument('--source-path', type=str, help='manifest files to use as source',
                             default='input/test_manifest.csv,input/test_manifest.csv')
    adda_parser.add_argument('--target-path', type=str, help='manifest files to use as target',
                             default='input/test_manifest.csv,input/test_manifest.csv')
    adda_parser.add_argument('--iterations', type=int, default=500)
    adda_parser.add_argument('--adda-epochs', type=int, default=5)
    adda_parser.add_argument('--k-disc', type=int, default=5)
    adda_parser.add_argument('--k-clf', type=int, default=10)
    return parser