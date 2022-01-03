//===================================================================
// File: Watchdog.hpp
//===================================================================
#pragma once

//-------------------------------------------------------------------
// Includes
//-------------------------------------------------------------------
#include <Arduino.h>

//-------------------------------------------------------------------
// Watchdog namespace
//-------------------------------------------------------------------
namespace watchdog {
    class Watchdog {
      public:
        //-----------------------------
        // @brief Constructor for the watchdog module.
        Watchdog() = default;

        //-----------------------------------------------------------
        // Watchdog module functions
        //-----------------------------------------------------------
        //-----------------------------
        // @brief Starts the watchdog.
        // @param timeout - The timeout for the watchdog.
        void begin(const uint32_t timeout) const;

        //-----------------------------
        // @brief Set the timeout.
        // @param timeout - The timeout for the watchdog.
        void set(const uint32_t timeout) const;

        //-----------------------------
        // @brief Reload the watchdog.
        void reload() const;

      private:
        //-----------------------------
        // @brief Validate the watchdog timeout.
        // @param timeout - The timeout for the watchdog.
        // @return bool - True if the timeout is accepted.
        bool is_wdg_timeout(const uint32_t timeout) const;
    };
} // namespace watchdog
