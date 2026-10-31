/* Read-only independent libavformat/libavcodec packet and frame count audit. */
#include <stdio.h>
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/error.h>

static int receive(AVCodecContext *ctx, AVFrame *frame, long *frames) {
    int ret;
    while ((ret = avcodec_receive_frame(ctx, frame)) >= 0) {
        ++*frames;
        av_frame_unref(frame);
    }
    return ret == AVERROR(EAGAIN) || ret == AVERROR_EOF ? 0 : ret;
}
int main(int argc, char **argv) {
    AVFormatContext *fmt = NULL; AVCodecContext *ctx = NULL;
    AVPacket *pkt = NULL; AVFrame *frame = NULL; int ret, stream, errors = 0;
    long packets = 0, frames = 0, no_vcl = 0, malformed = 0;
    int nal_len = 4;
    if (argc != 2) { fprintf(stderr, "usage: video_decode_audit video\n"); return 2; }
    if ((ret = avformat_open_input(&fmt, argv[1], NULL, NULL)) < 0 ||
        (ret = avformat_find_stream_info(fmt, NULL)) < 0) return 3;
    stream = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (stream < 0) return 4;
    AVStream *st = fmt->streams[stream];
    const AVCodec *decoder = avcodec_find_decoder(st->codecpar->codec_id);
    if (!decoder || !(ctx = avcodec_alloc_context3(decoder))) return 5;
    if (avcodec_parameters_to_context(ctx, st->codecpar) < 0 || avcodec_open2(ctx, decoder, NULL) < 0) return 6;
    if (st->codecpar->codec_id == AV_CODEC_ID_H264 && st->codecpar->extradata_size >= 5)
        nal_len = (st->codecpar->extradata[4] & 3) + 1;
    printf("nal_length_bytes=%d\n", nal_len);
    pkt = av_packet_alloc(); frame = av_frame_alloc();
    if (!pkt || !frame) return 7;
    printf("stream_metadata_nb_frames=%" PRId64 "\ncodec=%s\ntime_base=%d/%d\navg_frame_rate=%d/%d\n",
           st->nb_frames, decoder->name, st->time_base.num, st->time_base.den, st->avg_frame_rate.num, st->avg_frame_rate.den);
    while ((ret = av_read_frame(fmt, pkt)) >= 0) {
        if (pkt->stream_index == stream) {
            packets++;
            if (st->codecpar->codec_id == AV_CODEC_ID_H264) {
                int pos = 0, vcl = 0, bad = 0;
                while (pos < pkt->size) {
                    if (pkt->size - pos < nal_len) { bad = 1; break; }
                    unsigned int size = 0;
                    for (int k = 0; k < nal_len; k++) size = (size << 8) | pkt->data[pos++];
                    if (size == 0 || size > (unsigned)(pkt->size - pos)) { bad = 1; break; }
                    int type = pkt->data[pos] & 31;
                    if (type >= 1 && type <= 5) vcl = 1;
                    pos += (int)size;
                }
                if (bad) malformed++;
                if (!vcl) { no_vcl++; printf("packet_without_vcl=%ld bytes=%d pts=%" PRId64 " malformed=%d\n", packets, pkt->size, pkt->pts, bad); }
            }
            int rc = avcodec_send_packet(ctx, pkt);
            if (rc < 0) { errors++; fprintf(stderr, "send_error packet=%ld code=%d\n", packets, rc); }
            else if ((rc = receive(ctx, frame, &frames)) < 0) { errors++; fprintf(stderr, "receive_error packet=%ld code=%d\n", packets, rc); }
        }
        av_packet_unref(pkt);
    }
    printf("read_end_code=%d eof=%d\n", ret, ret == AVERROR_EOF);
    if (ret != AVERROR_EOF) errors++;
    ret = avcodec_send_packet(ctx, NULL);
    if (ret < 0 || receive(ctx, frame, &frames) < 0) { errors++; fprintf(stderr, "flush_error=%d\n", ret); }
    printf("video_packets=%ld\ndecoded_frames=%ld\npackets_without_vcl=%ld\nmalformed_nal_packets=%ld\ndecode_errors=%d\n", packets, frames, no_vcl, malformed, errors);
    av_frame_free(&frame); av_packet_free(&pkt); avcodec_free_context(&ctx); avformat_close_input(&fmt);
    return errors ? 1 : 0;
}
