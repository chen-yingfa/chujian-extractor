from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List



def iter_files(data_dir: Path):
    for dir in data_dir.iterdir():
        if dir.is_dir():
            for file in dir.iterdir():
                if not file.name.endswith('_1.png'):
                    yield file


def extract_modern(img: Image.Image, file: Path):
    '''Extract the modern Chinese text in the image, and save to file.'''
    raise NotImplementedError


def get_luminance(arr: np.array) -> np.array:
    '''Get the luminance of an image.'''
    return np.mean(arr, axis=-1)


def smooth_vec(vec: np.array, window_width: int, pad: int = 1) -> np.array:
    '''Smooth a vector by averaging over a window.'''
    padded = np.pad(
        vec, 
        (window_width // 2, window_width // 2), 
        'constant', 
        constant_values=(vec[0], vec[-1]),
    )
    cumsum_vec = np.cumsum(padded)
    smoothed = (cumsum_vec[window_width:] - cumsum_vec[:-window_width])
    return smoothed / window_width


def crop_rows(arr: np.array, threshold: float, padding: int) -> np.array:
    '''
    Crop rows (vertically) to remove white padding.
    Assuming it is 3D array (H, W, C).
    '''
    row_mean = np.mean(arr, axis=(1, 2)) / 255
    indices = np.where(row_mean < threshold)[0]
    lo, hi = indices[0], indices[-1]
    return arr[lo - padding : hi + padding]


def handle_split_slips(arr: np.array, padding: int) -> List[np.array]:
    '''
    Handle the case where a slip is split into multiple parts.
    '''
    Image.fromarray(arr).save('split_slip.png')

    # Just check if there is white rows in the middle of the image.
    row_mean = np.mean(arr, axis=(1, 2)) / 255
    white_rows = np.where(row_mean > 0.99)[0]
    REMOVE_PADDING = 6
    white_rows = white_rows[white_rows > REMOVE_PADDING]
    white_rows = white_rows[white_rows < len(row_mean) - REMOVE_PADDING]
    if len(white_rows) == 0:
        return [arr]
    assert len(white_rows) != 0

    # Split the slip into multiple parts
    print('Splitting slip...')
    white_rows = np.insert(white_rows, 0, -1)
    white_rows = np.append(white_rows, len(arr))
    # print(white_rows, len(white_rows))
    i = 0
    splits = []
    while i + 1 < len(white_rows):
        top = max(0, white_rows[i] + 1 - padding)
        bottom = min(len(arr), white_rows[i + 1] + padding)
        splits.append(arr[top : bottom])
        Image.fromarray(splits[-1]).save(f'split_{top}_{bottom}.png')

        # Search for next occurence of non-white row
        i += 1
        while (
            i + 1 < len(white_rows) and 
            white_rows[i] + 1 == white_rows[i + 1]
        ):
            i += 1
    return splits


def extract_bslips_and_texts(arr: np.array) -> List[np.array]:
    '''
    Extract the bamboo strip and printed modern text from the image.

    Assume that `arr` is in RGB.
    '''
    PADDING = 4
    MAX_ROW_LUMIN = 0.99

    # Crop away top and bottom white padding
    arr = crop_rows(arr, MAX_ROW_LUMIN, 4)
    Image.fromarray(arr).save('cropped.png')

    col_lumins = np.mean(arr, axis=(0, 2)) / 255
    col_lumins = smooth_vec(col_lumins, 8)
    plt.plot(col_lumins, linewidth=0.5)
    plt.savefig('col_lumins.png')
    plt.clf()

    # Scan left to right, and extract slips
    print('Extracting slips...')
    MAX_COL_LUMIN = 0.98
    lo = 0
    bslips = []
    texts = []
    while lo < len(col_lumins):
        if col_lumins[lo] < MAX_COL_LUMIN:
            hi = lo + 1
            while hi < len(col_lumins) and col_lumins[hi] < MAX_COL_LUMIN:
                hi += 1
            # Crop a vertical slip
            print(f'Found a slip: {lo} to {hi}')
            slip_arr = arr[:, lo - PADDING: hi + PADDING, :]
            slip_arr = crop_rows(slip_arr, 0.96, 4)
            if is_bslip(slip_arr):
                bslips += handle_split_slips(slip_arr, PADDING)[::-1]
            else:
                texts.append(slip_arr)
            lo = hi
        else:
            lo += 1
    bslips = bslips[::-1]  # Reverse, because Chujian ordered right to left
    texts = texts[::-1]
    return bslips, texts



def extract_bslip_glyphs(arr: np.array) -> List[np.array]:
    '''
    Extract all Chujian glyphs from an image of a bamboo slip.
    '''
    # Crop fixed height from top and bottom
    CROP_HEIGHT = 20

    arr = arr[CROP_HEIGHT:-CROP_HEIGHT]

    # Make black carvings more black
    processed = arr / 255
    processed = np.sqrt(processed)
    processed = processed / np.max(processed)  # Normalize

    row_lumin = np.mean(processed, axis=(1, 2))
    row_lumin = smooth_vec(row_lumin, 32)
    plt.plot(row_lumin, linewidth=0.5)
    plt.savefig('row_lumins.png')
    plt.clf()

    diff = np.gradient(row_lumin)
    diff = smooth_vec(diff, 32)
    diff /= np.max(diff)  # Normalize
    plt.plot(diff, linewidth=0.5)
    plt.savefig('diff.png')
    plt.clf()

    Image.fromarray(arr).save('bslip.png')

    # Scan top to bottom, and extract glyphs
    MAX_GLYPH_HEIGHT = 120
    MIN_GLYPH_HEIGHT = 8
    MIN_GLYPH_GRAD_DIFF = 0.18
    MAX_START_GRAD = -0.1
    MIN_GLYPH_GAP = 16
    SEARCH_WINDOW = 16
    PADDING = 12

    # print('Extracting glyphs...')
    lo = 0
    glyphs = []
    while lo < len(diff):
        if diff[lo] < MAX_START_GRAD:
            # Search for first local minimum gradient
            # Short peek for possible local min
            while (
                lo + SEARCH_WINDOW < len(diff) and 
                diff[lo] > np.min(diff[lo + 1:lo + SEARCH_WINDOW])
            ):
                lo += 1 + np.argmin(diff[lo+1 : lo + 6])

            # Search for next local maximium gradient
            min_hi = lo + MIN_GLYPH_HEIGHT
            if min_hi > len(diff): break  # Reached end
            max_hi = lo + MAX_GLYPH_HEIGHT
            hi = min_hi + np.argmax(diff[min_hi : max_hi])
            grad_diff = diff[hi] - diff[lo]
            if grad_diff > MIN_GLYPH_GRAD_DIFF:
                # Found glyph, extract it
                top = max(0, lo - PADDING)
                bottom = min(len(diff), hi + PADDING)
                # print(f'Found a glyph: {top} to {bottom}', grad_diff)
                glyph_arr = arr[top : bottom]
                glyphs.append(glyph_arr)
            lo = max(lo + SEARCH_WINDOW, hi + MIN_GLYPH_GAP)
        else:
            lo += 1
    return glyphs


def is_bslip(arr: np.array) -> bool:
    '''Check if an image is a bamboo slip.'''
    # Heuristic: Get the average whiteness of the image.
    # but ignore rows that are completely white, because there are white gaps
    # between slips.
    row_mean = np.mean(arr, axis=(1, 2)) / 255
    row_mean = row_mean[row_mean < 0.99]  # Ignore complete white rows
    white_prop = np.mean(row_mean)
    return white_prop < 0.65


def extract_chujian(img: Image.Image, result_dir: Path):
    '''
    Extract all characters sequentially (right to left and top to bottom) from
    an image of multiple Chujian bamboo slips.
    '''
    arr = np.array(img)
    bslips, texts = extract_bslips_and_texts(arr)
    for bslip_i, bslip in enumerate(bslips):
        file = result_dir / f'bslip_{bslip_i}.png'
        # print(f'Extracted {file}, shape: {bslip.shape}')
        Image.fromarray(bslip).save(file)

        if is_bslip(bslip):
            print(f'Extracting glyphs from {file}...')
            glyphs = extract_bslip_glyphs(bslip)
            print(f'Found {len(glyphs)} glyphs')
            for glyph_i, glyph in enumerate(glyphs):
                file = result_dir / f'glyph_{bslip_i}_{glyph_i}.png'
                # print(f'Extracted {file}, shape: {glyph.shape}')
                Image.fromarray(glyph).save(file)

    for text_i, text in enumerate(texts):
        file = result_dir / f'text_{text_i}.png'
        Image.fromarray(text).save(file)


if __name__ == '__main__':
    DATA_DIR = Path(r'D:\donny\code\ml\chujian\data\andajian1_image')
    RESULT_DIR = Path('result')
    files = iter_files(DATA_DIR)
    for file in files:
        if file.name.endswith('_10.png'):
            continue
        print(file)

        result_dir = RESULT_DIR / file.parent.name / file.stem
        result_dir.mkdir(exist_ok=True, parents=True)
        img = Image.open(file).convert('RGB')
        print(img)
        img.save(result_dir / 'image.png')
        extract_chujian(img, result_dir)

        exit()