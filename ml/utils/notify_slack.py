import argparse
import requests
import json


def slack_args(parser):
    parser.add_argument('--body', '-t', default='notification',
                        help='notification body')
    parser.add_argument('--webhook-url', help='which address to send')
    return parser


def notify_slack(cfg):
    data = json.dumps({'text': cfg['body']})
    requests.post(cfg['webhook_url'], data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='slack notification')
    expt_conf = vars(slack_args(parser).parse_args())
