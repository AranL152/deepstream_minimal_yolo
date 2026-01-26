#include <gst/gst.h>
#include <glib.h>

#include <iostream>
#include <string>
#include <fstream>
#include <cctype>
#include <iomanip>

// DeepStream meta helpers
#include "gstnvdsmeta.h"
#include "nvdsmeta.h"

static bool running_in_wsl() {
    if (g_getenv("WSL_INTEROP") != NULL) return true;

    std::ifstream f("/proc/sys/kernel/osrelease");
    if (!f.good()) return false;
    std::string s;
    std::getline(f, s);
    for (char &c : s) c = (char)std::tolower((unsigned char)c);
    return s.find("microsoft") != std::string::npos;
}

static void wsl_prefer_software_decode() {
    // Must run BEFORE gst_init().
    if (!running_in_wsl()) return;

    // If user already set it in shell, don't override.
    if (g_getenv("GST_PLUGIN_FEATURE_RANK") != NULL) return;

    // Down-rank NVIDIA decoders that often fail in WSL (/dev/nvidia0 issues).
    g_setenv("GST_PLUGIN_FEATURE_RANK", "nvh264dec:0,nvh265dec:0,nvv4l2decoder:0", FALSE);
}

static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data) {
    (void)bus;
    GMainLoop *loop = (GMainLoop *)data;

    switch (GST_MESSAGE_TYPE(msg)) {
    case GST_MESSAGE_EOS:
        g_print("EOS received - stopping pipeline...\n");
        g_main_loop_quit(loop);
        break;

    case GST_MESSAGE_ERROR: {
        GError *err = NULL;
        gchar *debug = NULL;
        gst_message_parse_error(msg, &err, &debug);
        g_printerr("ERROR: %s\n", err ? err->message : "(unknown)");
        if (debug) g_printerr("Debug: %s\n", debug);
        g_clear_error(&err);
        g_free(debug);
        g_main_loop_quit(loop);
        break;
    }

    default:
        break;
    }
    return TRUE;
}

// uridecodebin creates pads dynamically; link ONLY raw video to our queue
static void on_pad_added(GstElement *src, GstPad *new_pad, gpointer data) {
    (void)src;
    GstElement *queue = (GstElement *)data;

    GstCaps *caps = gst_pad_get_current_caps(new_pad);
    if (!caps) caps = gst_pad_query_caps(new_pad, NULL);
    if (!caps) return;

    GstStructure *str = gst_caps_get_structure(caps, 0);
    const gchar *name = gst_structure_get_name(str);

    // Only accept decoded raw video
    if (!g_str_has_prefix(name, "video/x-raw")) {
        gst_caps_unref(caps);
        return;
    }

    GstPad *sink_pad = gst_element_get_static_pad(queue, "sink");
    if (!sink_pad) {
        gst_caps_unref(caps);
        return;
    }

    if (gst_pad_is_linked(sink_pad)) {
        gst_object_unref(sink_pad);
        gst_caps_unref(caps);
        return;
    }

    GstPadLinkReturn ret = gst_pad_link(new_pad, sink_pad);
    if (GST_PAD_LINK_FAILED(ret)) {
        g_printerr("Failed to link decodebin pad (%s) to queue (ret=%d)\n", name, ret);
    } else {
        g_print("decodebin pad-linked: %s\n", name);
    }

    gst_object_unref(sink_pad);
    gst_caps_unref(caps);
}

// Probe at OSD sink pad: print detections (label, confidence, bbox)
static GstPadProbeReturn osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer /*u_data*/) {
    (void)pad;

    GstBuffer *buf = (GstBuffer *)info->data;
    if (!buf) return GST_PAD_PROBE_OK;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta) return GST_PAD_PROBE_OK;

    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
        if (!frame_meta) continue;

        // Count objects first for a nice frame header
        int obj_total = 0;
        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)(l_obj->data);
            if (obj_meta) obj_total++;
        }

        // If you want a line per frame even when empty, change this.
        if (obj_total == 0) continue;

        std::cout << "\n--- Frame " << frame_meta->frame_num
                  << ": " << obj_total << " objects detected ---\n";

        int idx = 0;
        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)(l_obj->data);
            if (!obj_meta) continue;
            idx++;

            const char *label = (obj_meta->obj_label && obj_meta->obj_label[0]) ? obj_meta->obj_label : "(no-label)";
            float conf = obj_meta->confidence; // typically 0..1

            NvOSD_RectParams &r = obj_meta->rect_params;
            float x1 = r.left;
            float y1 = r.top;
            float x2 = r.left + r.width;
            float y2 = r.top + r.height;

            std::cout << "  "
                      << std::setw(2) << idx << ") "
                      << "Class: " << std::left << std::setw(16) << label
                      << " | Confidence: " << std::right << std::fixed << std::setprecision(2)
                      << (conf * 100.0f) << "% "
                      << " | BBox: [" << (int)x1 << ", " << (int)y1 << ", " << (int)x2 << ", " << (int)y2 << "]\n";
        }
    }

    return GST_PAD_PROBE_OK;
}

int main(int argc, char *argv[]) {
    // Must happen before gst_init() so decodebin ranks are applied early.
    wsl_prefer_software_decode();
    gst_init(&argc, &argv);

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.mp4> <pgie_config.txt>\n";
        return 1;
    }

    std::string input_path = argv[1];
    std::string pgie_config = argv[2];

    std::cout << "=== Minimal YOLO Detection App (display + terminal output) ===\n"
              << "Video:  " << input_path << "\n"
              << "Config: " << pgie_config << "\n\n";

    // Build a proper file:// URI (handles spaces)
    gchar *uri = gst_filename_to_uri(input_path.c_str(), NULL);
    if (!uri) {
        std::cerr << "Failed to create URI for input file.\n";
        return 1;
    }

    // Elements
    GstElement *pipeline   = gst_pipeline_new("yolo-pipeline");
    GstElement *source     = gst_element_factory_make("uridecodebin", "source");
    GstElement *q_pre      = gst_element_factory_make("queue", "q_pre");

    GstElement *cpu_conv0  = gst_element_factory_make("videoconvert", "cpu_conv0");
    GstElement *scale      = gst_element_factory_make("videoscale", "scale");
    GstElement *rate       = gst_element_factory_make("videorate", "rate");
    GstElement *caps_cpu   = gst_element_factory_make("capsfilter", "caps_cpu");

    GstElement *nvconv_pre = gst_element_factory_make("nvvideoconvert", "nvconv_pre");
    GstElement *caps_nvmm  = gst_element_factory_make("capsfilter", "caps_nvmm");

    GstElement *streammux  = gst_element_factory_make("nvstreammux", "streammux");
    GstElement *pgie       = gst_element_factory_make("nvinfer", "primary-nvinference-engine");

    GstElement *nvconv_osd = gst_element_factory_make("nvvideoconvert", "nvconv_osd");
    GstElement *caps_osd   = gst_element_factory_make("capsfilter", "caps_osd");
    GstElement *nvosd      = gst_element_factory_make("nvdsosd", "nvosd");

    GstElement *nvconv_disp= gst_element_factory_make("nvvideoconvert", "nvconv_disp");
    GstElement *caps_disp  = gst_element_factory_make("capsfilter", "caps_disp");
    GstElement *cpu_conv1  = gst_element_factory_make("videoconvert", "cpu_conv1");
    GstElement *sink       = gst_element_factory_make("ximagesink", "x11_sink");

    if (!pipeline || !source || !q_pre ||
        !cpu_conv0 || !scale || !rate || !caps_cpu ||
        !nvconv_pre || !caps_nvmm ||
        !streammux || !pgie ||
        !nvconv_osd || !caps_osd || !nvosd ||
        !nvconv_disp || !caps_disp || !cpu_conv1 || !sink) {
        g_printerr("ERROR: One or more elements could not be created\n");
        g_free(uri);
        return 1;
    }

    // Properties
    g_object_set(G_OBJECT(source), "uri", uri, NULL);
    g_object_set(G_OBJECT(pgie), "config-file-path", pgie_config.c_str(), NULL);

    // Keep playback real-time (don’t run as fast as possible)
    g_object_set(G_OBJECT(sink), "sync", TRUE, NULL);

    // (Optional) if your file plays too fast/slow because of weird timestamps,
    // you can force a rate here. Otherwise, you can remove framerate=... below.
    GstCaps *caps_cpu_val = gst_caps_from_string("video/x-raw,width=1920,height=1080,framerate=25/1");
    g_object_set(G_OBJECT(caps_cpu), "caps", caps_cpu_val, NULL);
    gst_caps_unref(caps_cpu_val);

    // Upload/convert into NVMM for nvstreammux
    GstCaps *caps_nvmm_val = gst_caps_from_string("video/x-raw(memory:NVMM),format=NV12,width=1920,height=1080");
    g_object_set(G_OBJECT(caps_nvmm), "caps", caps_nvmm_val, NULL);
    gst_caps_unref(caps_nvmm_val);

    // NVMM RGBA before OSD
    GstCaps *caps_osd_val = gst_caps_from_string("video/x-raw(memory:NVMM),format=RGBA");
    g_object_set(G_OBJECT(caps_osd), "caps", caps_osd_val, NULL);
    gst_caps_unref(caps_osd_val);

    // CPU RGBA for ximagesink
    GstCaps *caps_disp_val = gst_caps_from_string("video/x-raw,format=RGBA");
    g_object_set(G_OBJECT(caps_disp), "caps", caps_disp_val, NULL);
    gst_caps_unref(caps_disp_val);

    g_object_set(G_OBJECT(streammux),
                 "batch-size", 1,
                 "width", 1920,
                 "height", 1080,
                 "batched-push-timeout", 4000000,
                 "live-source", 0,
                 NULL);

    // Add to pipeline
    gst_bin_add_many(GST_BIN(pipeline),
                     source, q_pre,
                     cpu_conv0, scale, rate, caps_cpu, nvconv_pre, caps_nvmm,
                     streammux, pgie,
                     nvconv_osd, caps_osd, nvosd,
                     nvconv_disp, caps_disp, cpu_conv1, sink,
                     NULL);

    // Link (decodebin -> q_pre is dynamic)
    if (!gst_element_link_many(q_pre, cpu_conv0, scale, rate, caps_cpu, nvconv_pre, caps_nvmm, NULL)) {
        g_printerr("ERROR: Failed to link pre-mux chain\n");
        g_free(uri);
        return 1;
    }

    // Link caps_nvmm -> streammux.sink_0 (request pad)
    GstPad *mux_sinkpad = gst_element_request_pad_simple(streammux, "sink_0");
    GstPad *srcpad      = gst_element_get_static_pad(caps_nvmm, "src");
    if (!mux_sinkpad || !srcpad) {
        g_printerr("ERROR: Failed to get pads for linking into streammux\n");
        if (mux_sinkpad) gst_object_unref(mux_sinkpad);
        if (srcpad) gst_object_unref(srcpad);
        g_free(uri);
        return 1;
    }
    if (gst_pad_link(srcpad, mux_sinkpad) != GST_PAD_LINK_OK) {
        g_printerr("ERROR: Failed to link caps_nvmm -> streammux sink_0\n");
        gst_object_unref(srcpad);
        gst_object_unref(mux_sinkpad);
        g_free(uri);
        return 1;
    }
    gst_object_unref(srcpad);
    gst_object_unref(mux_sinkpad);

    // Downstream
    if (!gst_element_link_many(streammux, pgie, nvconv_osd, caps_osd, nvosd,
                               nvconv_disp, caps_disp, cpu_conv1, sink, NULL)) {
        g_printerr("ERROR: Failed to link streammux -> ... -> sink\n");
        g_free(uri);
        return 1;
    }

    // Hook up decodebin pad-added to q_pre
    g_signal_connect(G_OBJECT(source), "pad-added", G_CALLBACK(on_pad_added), q_pre);

    // Add OSD probe for terminal output
    GstPad *osd_sink_pad = gst_element_get_static_pad(nvosd, "sink");
    if (osd_sink_pad) {
        gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
                          osd_sink_pad_buffer_probe, NULL, NULL);
        gst_object_unref(osd_sink_pad);
    } else {
        g_printerr("WARNING: Could not get nvosd sink pad; no terminal output will be printed\n");
    }

    // Bus + mainloop
    GMainLoop *loop = g_main_loop_new(NULL, FALSE);
    GstBus *bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    gst_bus_add_watch(bus, bus_call, loop);
    gst_object_unref(bus);

    g_print("Starting pipeline...\n");
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    g_main_loop_run(loop);

    g_print("Cleaning up...\n");
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(GST_OBJECT(pipeline));
    g_main_loop_unref(loop);

    g_free(uri);
    return 0;
}

