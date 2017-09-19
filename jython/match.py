#!/usr/bin/env python

import fire
import re
from collections import Counter

"""
Provide
- pattern name 
- quoted pattern (no name, contains parentheses)
- quoted named pattern (name;pattern)
- tab delimited pattern file
- semicolon delimited pattern file
"""

default_pattern_name = 'match'

patterns = {
		'pool0': 'GGTCTCCACCG(.*)GTTT.GAGACG(.*)CGTCTC.TTCC(.*)ACTGC',
		'pool1': 'CCACCG(.*)GTTT..GTCTTC(.*)GAAGAC..TTCC(.*)ACTGGC' 
           }


def parse_semicolon_pattern(s):
	a, b = s.split(';')
	return a, b

def parse_pattern_name(s):
	return s, patterns[s]

def parse_pattern(s):
	re.compile(s)
	return default_pattern_name, s




def globular(f):
	def wrapped(self, *args, **kwargs):
		[f(x, **kwargs) for x in args]
	return wrapped

class Matcher(object):
	"""Matches input against regular expression, outputs capture groups."""

	@staticmethod
	@globular
	def match(filename, pattern_name='pool1'):
		""" Matches input against pattern, output histogram table"""
		output = []
		i = 0
		with open(filename, 'r') as fh:
			for line in fh:
				match = re.findall(patterns[pattern_name], line)
				if match:
					output.append(tuple(match[0]))
				i += 1

		hist = Counter(output)

		output_filename = filename + '.match.%s' % pattern_name
		output_txt = '\n'.join('\t'.join(line) for line in output)
		with open(output_filename, 'w') as fh:
			fh.write(output_txt)

		hist_filename = filename + '.match.%s.hist' % pattern_name
		hist_sorted = sorted(hist.items(), key=lambda x: -x[1])
		hist_txt = '\n'.join('%d\t%s' % (v, '\t'.join(k)) for k,v in hist_sorted)
		with open(hist_filename, 'w') as fh:
			fh.write(hist_txt)

		print '%d matches / %d lines written to %s' % (len(output), i, output_filename)
		print '%d unique matches written to %s' % (len(hist), hist_filename)


if __name__ == '__main__':
  fire.Fire(Matcher)
