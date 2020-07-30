#!/usr/bin/env python3
from PIL import Image, ImageFilter, ImageColor
from pprint import pprint
from sys import argv
from math import *

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

def sobel(leftSobel_pixels, topSobel_pixels):
	w, h = len(leftSobel_pixels[0]), len(leftSobel_pixels)
	return [[(leftSobel_pixels[y][x]**2 + topSobel_pixels[y][x]**2)**.5 for x in range(w)] for y in range(h)]

def sobelDirection(leftSobel_pixels, topSobel_pixels, sobel_pixels):
	def _sobelDirection(x, y):
		angle = atan(leftSobel_pixels[y][x] / (topSobel_pixels[y][x] if topSobel_pixels[y][x] != 0 else 1e-9))
		hue = (angle * 180/pi) % 360
		value = sobel_pixels[y][x] / 255
		return ImageColor.getrgb(f'hsv({hue},100%,{value:%})')
	w, h = len(leftSobel_pixels[0]), len(leftSobel_pixels)
	return [[_sobelDirection(x, y) for x in range(w)] for y in range(h)]



name = argv[1].partition('.')[0]
original = Image.open(argv[1])


luminance = original.convert('L')
luminance_pixels = getpixels(luminance)
luminance_pixels_duplicated = duplicate(luminance_pixels)
getimg(luminance_pixels).save(f'{name}_luminance.png') # destroy metadata

blurred = original.filter(ImageFilter.GaussianBlur(radius=2))
blurred.save(f'{name}_gaussianBlur.png')


leftSobel_pixels = applyKernel(luminance_pixels_duplicated, LEFT_SOBEL, 1)
signedimg(leftSobel_pixels).save(f'{name}_leftSobel.png')

topSobel_pixels = applyKernel(luminance_pixels_duplicated, TOP_SOBEL, 1)
signedimg(topSobel_pixels).save(f'{name}_topSobel.png')


sobel_pixels = sobel(leftSobel_pixels, topSobel_pixels)
sobel = getimg(sobel_pixels)
sobel.save(f'{name}_sobel.png')


sobelDirection_pixels = sobelDirection(leftSobel_pixels, topSobel_pixels, sobel_pixels)
getimg(sobelDirection_pixels).save(f'{name}_sobelDirection.png')

Image.composite(getimg([[0 for x in range(original.width)] for y in range(original.height)]), original, sobel).save(f'{name}_edgesRemoved.png')

Image.composite(blurred, original, sobel).save(f'{name}_aa.png')

