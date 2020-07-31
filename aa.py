#!/usr/bin/env python3
from PIL import Image, ImageFilter, ImageColor
from pprint import pprint
from sys import argv
from math import *
from numpy import linspace
from matplotlib import pyplot as plt



def getpixels(img):
	w, h = img.size
	data = img.getdata()
	return [[data[x + y * w] for x in range(w)] for y in range(h)]

def getimg(pixels):
	w, h = len(pixels[0]), len(pixels)
	model = 'L' if isinstance(pixels[0][0], (int, float)) else 'RGB'
	img = Image.new(model, (w, h))
	for y in range(h):
		for x in range(w):
			img.putpixel((x, y), round(pixels[y][x]) if model == 'L' else tuple(round(d) for d in pixels[y][x]))
	return img

def signedimg(pixels):
	w, h = len(pixels[0]), len(pixels)
	img = Image.new('RGB', (w, h))
	for y in range(h):
		for x in range(w):
			img.putpixel((x, y), (round(pixels[y][x]), 0, 0) if pixels[y][x] > 0 else (0, round(-pixels[y][x]), 0))
	return img

def duplicate(pixels):
	w, h = len(pixels[0]), len(pixels)
	res = [[d for d in row] for row in pixels]
	res = [[row[0]] + row + [row[-1]] for row in res]
	res.insert(0, res[0])
	res.insert(len(res), res[-1])
	return res



def kernelOffsets(n=3):
	return [(x - n//2, y - n//2) for x in range(n) for y in range(n)]

def kernelScale(kernel):
	normal_scale = pos_scale = neg_scale = 0
	for row in kernel:
		for k in row:
			normal_scale += k
			if k > 0:
				pos_scale += k
			else:
				neg_scale -= k
	return normal_scale if normal_scale != 0 else max(pos_scale, neg_scale)

def applyKernel(pixels, kernel, scale=None):
	if scale == None:
		scale = kernelScale(kernel)
	n = len(kernel[0])
	offset = n // 2
	w, h = len(pixels[0]), len(pixels)
	res = [[0 for x in range(w - 2*offset)] for y in range(h - 2*offset)]
	for y in range(h - 2*offset):
		for x in range(w - 2*offset):
			d = 0
			for (dx, dy) in kernelOffsets(n):
				d += kernel[dy][dx] * pixels[y + offset + dy][x + offset + dx]
			res[y][x] = d / scale
	return res

LEFT_SOBEL = [[1, 0, -1],
				 [2, 0, -2],
				 [1, 0, -1]]
TOP_SOBEL = [[ 1,  2,  1],
				[ 0,  0,  0],
				[-1, -2, -1]]


def sobelMerge(leftSobel_pixels, topSobel_pixels):
	w, h = len(leftSobel_pixels[0]), len(leftSobel_pixels)
	return [[((leftSobel_pixels[y][x] ** 2 + topSobel_pixels[y][x] ** 2) / 2) ** .5 for x in range(w)] for y in range(h)]

def sobelDirection(leftSobel_pixels, topSobel_pixels, sobel_pixels):
	def _sobelDirection(x, y):
		'''
			720: angleFromCoords(x, y) * 360/pi
			360: angleFromCoords(x, y) * 180/pi
			360x2 (normal): atan(x, y) * 360/pi
			180x2: atan(x, y) * 180/pi
			180x2-fixed: atan(x, abs(y)) * 180/pi
		'''
		#angle = angleFromCoords(leftSobel_pixels[y][x], topSobel_pixels[y][x] or 1e-9)
		#angle = atan(leftSobel_pixels[y][x] / (topSobel_pixels[y][x] or 1e-9))
		angle = atan(leftSobel_pixels[y][x] / abs(topSobel_pixels[y][x] or 1e-9))
		#hue = (angle * 360/pi) % 360
		hue = (angle * 180/pi) % 360
		value = sobel_pixels[y][x] / 255
		return ImageColor.getrgb(f'hsv({hue},100%,{value:%})')
	w, h = len(leftSobel_pixels[0]), len(leftSobel_pixels)
	return [[_sobelDirection(x, y) for x in range(w)] for y in range(h)]

def sign(x):
	return copysign(1, x)

def angleFromCoords(x, y):
	c = (x**2 + y**2)**.5
	x = min(max(-1, x / c), 1)
	y = min(max(-1, y / c), 1)
	return sign(asin(y)) * (acos(x)-pi) + pi


def total(pixels):
	res = 0
	for row in pixels:
		for d in row:
			res += d
	return res

def matchGamma(src, target):
	w, h = len(target[0]), len(target)
	target = [[min(target[y][x] / 255, 1) for x in range(w)] for y in range(h)]
	src_total = round(total(src) / 255)
	g = 5
	step = g / 2
	while True:
		curr = [[target[y][x] ** g for x in range(w)] for y in range(h)]
		curr_total = round(total(curr))
		if curr_total < src_total:
			g -= step
		elif curr_total > src_total:
			g += step
		else:
			break
		step /= 2
	return [[curr[y][x] * 255 for x in range(w)] for y in range(h)]



name = argv[1].partition('.')[0]
original = Image.open(argv[1])


luminance = original.convert('L')
luminance_pixels = getpixels(luminance)
luminance_pixels_duplicated = duplicate(luminance_pixels)
#getimg(luminance_pixels).save(f'{name}_luminance.png') # destroy metadata

leftSobel_pixels = applyKernel(luminance_pixels_duplicated, LEFT_SOBEL, 1)
topSobel_pixels = applyKernel(luminance_pixels_duplicated, TOP_SOBEL, 1)

sobel_pixels = sobelMerge(leftSobel_pixels, topSobel_pixels)
sobel = getimg(sobel_pixels)
sobel.save(f'{name}_sobel.png')


blurred = original.filter(ImageFilter.GaussianBlur(radius=2))
blurred_pixels_duplicated = duplicate(getpixels(blurred.convert('L')))
#blurred.save(f'{name}_blurred.png')

blurred_leftSobel_pixels = applyKernel(blurred_pixels_duplicated, LEFT_SOBEL, 1)
blurred_topSobel_pixels = applyKernel(blurred_pixels_duplicated, TOP_SOBEL, 1)
#signedimg(blurred_leftSobel_pixels).save(f'{name}_leftSobelBlurred.png')
#signedimg(blurred_topSobel_pixels).save(f'{name}_topSobelBlurred.png')

blurred_sobel_pixels = sobelMerge(blurred_leftSobel_pixels, blurred_topSobel_pixels)
blurred_sobel_pixels = matchGamma(sobel_pixels, blurred_sobel_pixels)
blurred_sobel = getimg(blurred_sobel_pixels)
blurred_sobel.save(f'{name}_sobelBlurred.png')


Image.composite(getimg([[0 for x in range(original.width)] for y in range(original.height)]), original, sobel).save(f'{name}_edgesRemoved.png')
Image.composite(getimg([[0 for x in range(original.width)] for y in range(original.height)]), original, blurred_sobel).save(f'{name}_edgesRemovedBlurred.png')

Image.composite(blurred, original, blurred_sobel).save(f'{name}_aa.png')

sobelDirection_pixels = sobelDirection(leftSobel_pixels, topSobel_pixels, sobel_pixels)
getimg(sobelDirection_pixels).save(f'{name}_sobelDirection.png')

blurred_sobelDirection_pixels = sobelDirection(blurred_leftSobel_pixels, blurred_topSobel_pixels, sobel_pixels)
getimg(blurred_sobelDirection_pixels).save(f'{name}_sobelDirectionBlurred.png')
