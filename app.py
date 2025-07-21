# app.py
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# notice proxy using. delete it if not need
#os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10809"
#os.environ["HTTP_PROXY"] = "http://127.0.0.1:10809"

import uuid
import re
import numpy as np
from flask import Flask, request, render_template, send_from_directory, make_response
from moviepy.editor import VideoFileClip
from pyannote.audio import Pipeline
from PIL import Image, ImageDraw, ImageEnhance
import torch

UPLOAD_FOLDER = 'uploads'
AVATAR_FOLDER = 'avatars'
OUTPUT_FOLDER = 'outputs'
STATIC_FOLDER = 'static'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AVATAR_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['AVATAR_FOLDER'] = AVATAR_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER

# 加载模型
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                    use_auth_token="Your Access Token")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline = pipeline.to(device)

# ---------------- 基础头像处理 ----------------
def make_circle_avatar(img_path, size=(200, 200)):
    img = Image.open(img_path).convert("RGBA").resize(size)
    mask = Image.new('L', size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, size[0], size[1]), fill=255)
    img.putalpha(mask)
    return img

def parse_num_speakers(form):
    raw = form.get("num_speakers", "")
    if raw is None:
        return 0
    raw = raw.strip()
    if not raw:
        return 0
    # 允许用户输入奇怪字符（比如“ 2 人 ”、“。3”），提取首个数字
    import re
    m = re.search(r'\d+', raw)
    if not m:
        return 0
    n = int(m.group())
    return n if n > 0 else 0

def load_avatars(video_basename):
    """仅加载当前视频的头像"""
    avatars = {}
    for fname in os.listdir(AVATAR_FOLDER):
        if not fname.lower().endswith(('png', 'jpg', 'jpeg')):
            continue
        if fname.startswith(video_basename + "_speaker_"):
            try:
                sid = int(fname.split('_')[-1].split('.')[0])
                avatars[sid] = make_circle_avatar(os.path.join(AVATAR_FOLDER, fname))
            except:
                pass
    return avatars

# ---------------- 原动态高度版本 ----------------
def generate_avatar_video(timeline, avatar_data, duration, fps=25):
    from moviepy.editor import VideoClip
    from PIL import Image

    avatar_size = (200, 200)

    # 预计算最大同时说话人数
    time_points = np.linspace(0, duration, int(duration * fps))
    max_concurrent = 1
    for t in time_points:
        speakers_now = set()
        for seg, _, label in timeline:
            if seg.start <= t <= seg.end:
                try:
                    sid = int(label.split('_')[-1])
                    if sid in avatar_data:
                        speakers_now.add(sid)
                except:
                    continue
        max_concurrent = max(max_concurrent, len(speakers_now))

    frame_width = avatar_size[0]
    frame_height = avatar_size[1] * max_concurrent

    def make_frame(t):
        speakers_on = []
        for seg, _, label in timeline:
            if seg.start <= t <= seg.end:
                try:
                    sid = int(label.split('_')[-1])
                    if sid in avatar_data:
                        speakers_on.append(sid)
                except:
                    continue

        bg_color = (0, 0, 0, 0)
        img = Image.new("RGBA", (frame_width, frame_height), bg_color)

        for i, sid in enumerate(sorted(speakers_on)):
            if i >= max_concurrent:
                break
            avatar = avatar_data[sid].resize(avatar_size)
            y_offset = i * avatar_size[1]
            img.paste(avatar, (0, y_offset), avatar)

        return np.array(img.convert("RGB")).astype(np.uint8)

    return VideoClip(make_frame, duration=duration).set_fps(fps).set_duration(duration)

# ---------------- 新的固定列版本 ----------------
def parse_speaker_id(label):
    if isinstance(label, str):
        m = re.search(r'(\d+)', label)
        if m:
            return int(m.group(1))
    return None

def build_speaker_intervals(timeline):
    speaker_intervals = {}
    for seg, _, label in timeline:
        sid = parse_speaker_id(label)
        if sid is None:
            continue
        speaker_intervals.setdefault(sid, []).append((seg.start, seg.end))
    for sid, ivs in speaker_intervals.items():
        ivs.sort()
        merged = []
        for s, e in ivs:
            if not merged or s > merged[-1][1]:
                merged.append([s, e])
            else:
                merged[-1][1] = max(merged[-1][1], e)
        speaker_intervals[sid] = [(a, b) for a, b in merged]
    return speaker_intervals

def generate_avatar_column_video(timeline, avatar_data, duration, fps=25,
                                 avatar_size=(200, 200), dim_factor=0.35,
                                 highlight_factor=1.0, bg_color=(0, 0, 0),
                                 smooth_ms=240):
    from moviepy.editor import VideoClip
    from PIL import Image

    speaker_intervals = build_speaker_intervals(timeline)
    all_ids = sorted(set(speaker_intervals.keys()) | set(avatar_data.keys()))
    if not all_ids:
        def empty_frame(t):
            return np.full((avatar_size[1], avatar_size[0], 3), bg_color, dtype=np.uint8)
        return VideoClip(empty_frame, duration=duration).set_fps(fps)

    norm_avatars = {}
    for sid in all_ids:
        if sid in avatar_data:
            base_img = avatar_data[sid].convert("RGBA").resize(avatar_size)
        else:
            base_img = Image.new("RGBA", avatar_size, (0, 0, 0, 0))
        norm_avatars[sid] = base_img

    bright_versions = {sid: ImageEnhance.Brightness(img).enhance(highlight_factor)
                       for sid, img in norm_avatars.items()}
    dim_versions = {sid: ImageEnhance.Brightness(img).enhance(dim_factor)
                    for sid, img in norm_avatars.items()}

    pad = smooth_ms / 1000.0
    padded = {sid: [(max(0.0, s - pad), min(duration, e + pad)) for s, e in ivs]
              for sid, ivs in speaker_intervals.items()}

    frame_width = avatar_size[0]
    frame_height = avatar_size[1] * len(all_ids)

    def is_active(sid, t):
        if sid not in padded:
            return False
        for s, e in padded[sid]:
            if s <= t <= e:
                return True
        return False

    def make_frame(t):
        frame = Image.new("RGBA", (frame_width, frame_height),
                          bg_color + (255,) if len(bg_color) == 3 else bg_color)
        for row, sid in enumerate(all_ids):
            top = row * avatar_size[1]
            active = is_active(sid, t)
            img_avatar = bright_versions[sid] if active else dim_versions[sid]
            frame.paste(img_avatar, (0, top), img_avatar)
        return np.array(frame.convert("RGB")).astype(np.uint8)

    return VideoClip(make_frame, duration=duration).set_fps(fps)

def generate_avatar_row_video(timeline, avatar_data, duration, fps=25,
                              avatar_size=(200, 200), dim_factor=0.35,
                              highlight_factor=1.0, bg_color=(0, 0, 0),
                              smooth_ms=240):
    from moviepy.editor import VideoClip
    from PIL import Image

    speaker_intervals = build_speaker_intervals(timeline)
    all_ids = sorted(set(speaker_intervals.keys()) | set(avatar_data.keys()))
    if not all_ids:
        def empty_frame(t):
            return np.full((avatar_size[1], avatar_size[0], 3), bg_color, dtype=np.uint8)
        return VideoClip(empty_frame, duration=duration).set_fps(fps)

    norm_avatars = {}
    for sid in all_ids:
        if sid in avatar_data:
            base_img = avatar_data[sid].convert("RGBA").resize(avatar_size)
        else:
            base_img = Image.new("RGBA", avatar_size, (0, 0, 0, 0))
        norm_avatars[sid] = base_img

    bright_versions = {sid: ImageEnhance.Brightness(img).enhance(highlight_factor)
                       for sid, img in norm_avatars.items()}
    dim_versions = {sid: ImageEnhance.Brightness(img).enhance(dim_factor)
                    for sid, img in norm_avatars.items()}

    pad = smooth_ms / 1000.0
    padded = {sid: [(max(0.0, s - pad), min(duration, e + pad)) for s, e in ivs]
              for sid, ivs in speaker_intervals.items()}

    frame_width = avatar_size[0] * len(all_ids)
    frame_height = avatar_size[1]

    def is_active(sid, t):
        if sid not in padded:
            return False
        for s, e in padded[sid]:
            if s <= t <= e:
                return True
        return False

    def make_frame(t):
        frame = Image.new("RGBA", (frame_width, frame_height),
                          bg_color + (255,) if len(bg_color) == 3 else bg_color)
        for col, sid in enumerate(all_ids):
            left = col * avatar_size[0]
            active = is_active(sid, t)
            img_avatar = bright_versions[sid] if active else dim_versions[sid]
            frame.paste(img_avatar, (left, 0), img_avatar)
        return np.array(frame.convert("RGB")).astype(np.uint8)

    return VideoClip(make_frame, duration=duration).set_fps(fps)

# ---------------- 路由 ----------------
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    lang = request.args.get('lang')

    # 如果 URL 没指定语言，就从 Cookie 里读
    if not lang:
        lang = request.cookies.get('lang', 'zh')
    if request.method == 'POST':
        video_file = request.files['video']
        video_filename = str(uuid.uuid4()) + '_' + video_file.filename
        video_path = os.path.join(UPLOAD_FOLDER, video_filename)
        file_ext = video_filename.split('.')[-1].lower()
        video_file.save(video_path)

        # 判断是否是 WAV 音频
        if file_ext == 'wav':
            wav_path = video_path
        else:
            # 提取音频
            wav_path = video_path.rsplit('.', 1)[0] + '.wav'
            clip = VideoFileClip(video_path)
            clip.audio.write_audiofile(wav_path, verbose=False, logger=None)

        num_speakers = parse_num_speakers(request.form)  # 默认0表示不指定
        print(str(num_speakers)+ 'speakers')
        if num_speakers > 0:
            diarization = pipeline(wav_path, num_speakers=num_speakers)
        else:
            diarization = pipeline(wav_path)

        # diarization = pipeline(wav_path)
        timeline = list(diarization.itertracks(yield_label=True))

        timeline_data_path = os.path.join(UPLOAD_FOLDER, f"{video_filename}.timeline.npy")
        np.save(timeline_data_path, timeline)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 2))
        for segment, _, label in timeline:
            start, end = segment.start, segment.end
            ax.barh(y=label, width=end - start, left=start)
        ax.set_xlabel("Time (s)")
        plt.tight_layout()

        timeline_img = os.path.join(app.config['STATIC_FOLDER'], f"{video_filename}.png")
        plt.savefig(timeline_img)
        plt.close()

        # return render_template("index.html",
        #                        timeline_path=os.path.basename(timeline_img),
        #                        allow_avatar_upload=True,
        #                        video_path=video_filename)

        speaker_ids = sorted({label for _, _, label in timeline})

        template_name = "index_en.html" if lang == 'en' else "index.html"
        resp = make_response(render_template(
            template_name,
            timeline_path=os.path.basename(timeline_img),
            allow_avatar_upload=True,
            video_path=video_filename,
            speaker_ids=speaker_ids,
            lang=lang
        ))

        # 设置 lang cookie，路径设置为根目录 '/'
        resp.set_cookie('lang', lang, max_age=60 * 60 * 24 * 365)  # 有效期1年
        return resp

    else:
        template_name = "index_en.html" if lang == 'en' else "index.html"
        resp = make_response(render_template(template_name, lang=lang))
        resp.set_cookie('lang', lang, max_age=60 * 60 * 24 * 365)
        return resp

from moviepy.editor import VideoFileClip, AudioFileClip

@app.route('/upload_avatars', methods=['POST'])
def upload_avatars():
    video_filename = request.form['video_path']
    video_basename = os.path.splitext(video_filename)[0]
    video_path = os.path.join(UPLOAD_FOLDER, video_filename)
    timeline_data_path = os.path.join(UPLOAD_FOLDER, f"{video_filename}.timeline.npy")
    timeline = np.load(timeline_data_path, allow_pickle=True).tolist()

    # 1. 删除原有头像
    for fname in os.listdir(AVATAR_FOLDER):
        if fname.startswith(video_basename + "_speaker_"):
            os.remove(os.path.join(AVATAR_FOLDER, fname))

    files = request.files.getlist("avatars")
    ids = request.form.getlist("avatar_speaker_ids")
    for f, sid in zip(files, ids):
        if f and sid.isdigit():
            save_path = os.path.join(AVATAR_FOLDER, f"{video_basename}_speaker_{sid}.png")
            f.save(save_path)

    avatar_data = load_avatars(video_basename)

    # 根据文件类型加载媒体
    ext = os.path.splitext(video_filename)[-1].lower()
    ext = os.path.splitext(video_filename)[-1].lower()
    if ext == ".wav":
        audio_clip = AudioFileClip(video_path)
        duration = audio_clip.duration
    else:
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio
        duration = video_clip.duration

    # 动态高度版本
    #dyn_clip = generate_avatar_video(timeline, avatar_data, duration)  #video input not set fps
    base_name = os.path.splitext(video_filename)[0]

    dyn_name = f"overlay_{base_name}.mp4"
    dyn_path = os.path.join(OUTPUT_FOLDER, dyn_name)
    dyn_clip = generate_avatar_video(timeline, avatar_data, duration).set_fps(25)
    dyn_clip = dyn_clip.set_audio(audio_clip)
    dyn_clip.write_videofile(dyn_path, codec="libx264", audio_codec="aac")

    column_name = f"column_overlay_{base_name}.mp4"
    column_path = os.path.join(OUTPUT_FOLDER, column_name)
    column_clip = generate_avatar_column_video(timeline, avatar_data, duration).set_fps(25)
    column_clip = column_clip.set_audio(audio_clip)
    column_clip.write_videofile(column_path, codec="libx264", audio_codec="aac")

    row_name = f"row_overlay_{base_name}.mp4"
    row_path = os.path.join(OUTPUT_FOLDER, row_name)
    row_clip = generate_avatar_row_video(timeline, avatar_data, duration).set_fps(25)
    row_clip = row_clip.set_audio(audio_clip)
    row_clip.write_videofile(row_path, codec="libx264", audio_codec="aac")

    speaker_ids = sorted({label for _, _, label in timeline})
    return render_template(
        "index.html",
        output_video=dyn_name,
        column_video=column_name,
        row_video=row_name,
        video_path=video_filename,
        speaker_ids=speaker_ids,
        allow_avatar_upload=True
    )




@app.route('/outputs/<filename>')
def output_video(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
