// llama_jinja.cpp
// Thin C-API wrapper around common_chat_templates_apply so C# (LlamaSharp)
// can use the full Jinja2 renderer from llama.cpp/common.

#include "llama.h"
#include "chat.h"

#include <string>
#include <vector>
#include <cstring>
#include <nlohmann/json_fwd.hpp>
#include "nlohmann/json.hpp"

#if defined(_WIN32)
#  define LLAMA_JINJA_API __declspec(dllexport)
#else
#  define LLAMA_JINJA_API __attribute__((visibility("default")))
#endif

extern "C" {

// Apply the Jinja2 chat template stored in the model (or override_template if non-empty).
//
// messages_json : JSON array of {"role":"user"|"assistant"|"system","content":"..."}
// add_generation_prompt : append the model-role generation prompt
// buf / buf_size        : output buffer; pass buf==NULL to query required size
//
// Returns number of bytes written (excluding NUL), or -1 on error.
LLAMA_JINJA_API int llama_jinja_apply_template(
        const struct llama_model * model,
        const char               * messages_json,
        bool                       add_generation_prompt,
        bool                       enable_thinking,
        char                     * buf,
        int                        buf_size)
{
    try {
        // Build the chat templates object from the model
        auto tmpls = common_chat_templates_init(model, "");

        // Parse messages from JSON
        auto j = nlohmann::json::parse(messages_json);

        std::vector<common_chat_msg> msgs;
        for (auto & m : j) {
            common_chat_msg msg;
            msg.role    = m.value("role", "user");
            msg.content = m.value("content", "");
            msgs.push_back(std::move(msg));
        }

        // Build inputs
        common_chat_templates_inputs inputs;
        inputs.messages                = msgs;
        inputs.add_generation_prompt   = add_generation_prompt;
        inputs.enable_thinking         = enable_thinking;
        inputs.use_jinja               = true;

        // Apply template
        auto result = common_chat_templates_apply(tmpls.get(), inputs);
        const std::string & prompt = result.prompt;

        if (!buf) {
            return (int)prompt.size();
        }

        int len = (int)prompt.size();
        if (len + 1 > buf_size) {
            // Buffer too small — write as much as fits
            memcpy(buf, prompt.c_str(), buf_size - 1);
            buf[buf_size - 1] = '\0';
            return len; // caller can reallocate and retry
        }

        memcpy(buf, prompt.c_str(), len + 1);
        return len;
    }
    catch (const std::exception & e) {
        fprintf(stderr, "[llama-jinja] exception: %s\n", e.what());
        return -1;
    }
    catch (...) {
        fprintf(stderr, "[llama-jinja] unknown exception\n");
        return -1;
    }
}

} // extern "C"
