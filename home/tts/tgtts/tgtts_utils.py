import sys
import os
import io
import time
import random
import math
import threading
import string
from requests_futures.sessions import FuturesSession
import requests


class ElapsedFuturesSession(FuturesSession):
	def request(self, method, url, hooks=None, *args, **kwargs):
		start = time.time()
		if hooks is None:
			hooks = {}

		def timing(r, *args, **kwargs):
			r.elapsed = time.time() - start
			r.overhead = None
			if 'X-Timing' in r.headers and float(r.headers['X-Timing']) > 0:
				r.overhead = time.time() - float(r.headers['X-Timing'])

		try:
			if isinstance(hooks['response'], (list, tuple)):
				# needs to be first so we don't time other hooks execution
				hooks['response'].insert(0, timing)
			else:
				hooks['response'] = [timing, hooks['response']]
		except KeyError:
			hooks['response'] = timing

		return super(ElapsedFuturesSession, self) \
			.request(method, url, hooks=hooks, *args, **kwargs)


def two_way_round(number, ndigits = 0, predigits = 0):
	number = round(number, ndigits)
	return (f"%0{predigits}.{ndigits}f") % number

def toms(number):
	return number*1000
