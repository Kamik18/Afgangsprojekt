//===================================================================
// File: Watchdog.cpp
//
// Description:
//  Module to encapsulation of watchdog functionality.
//===================================================================
//-------------------------------------------------------------------
// Includes
//-------------------------------------------------------------------
#include "Watchdog.hpp"
#include "stm32wbxx_ll_iwdg.h"
#include "stm32wbxx_ll_rcc.h"

namespace watchdog {
    //-----------------------------
    // @brief Starts the watchdog.
    // @param timeout - The timeout for the watchdog.
    void Watchdog::begin(const uint32_t timeout) const {
        if (is_wdg_timeout(timeout)) {
            // Enable the IWDG by writing 0x0000 CCCC in the IWDG_KR register
            ::LL_IWDG_Enable(IWDG);
            // Set timeout
            this->set(timeout);
        }
    }

    //-----------------------------
    // @brief Set the timeout.
    // @param timeout - The timeout for the watchdog.
    void Watchdog::set(const uint32_t timeout) const {
        // Compute the prescaler value
        uint8_t        prescaler = 0;
        uint16_t       divider   = 0;
        const uint16_t bitshift  = 4;

        // Convert timeout to seconds
        const uint32_t second = 1'000'000;
        const auto     t_sec  = static_cast<float_t>((timeout / second) * LSI_VALUE);

        do {
            divider = static_cast<uint16_t>(bitshift << prescaler);
            ++prescaler;
        } while ((t_sec / divider) > IWDG_RLR_RL);

        // Validate the prescaler
        --prescaler;
        if ((prescaler <= LL_IWDG_PRESCALER_256) && (is_wdg_timeout(timeout))) {
            const auto reload = static_cast<uint32_t>((t_sec / divider) - 1);

            // Enable register access by writing 0x0000 5555 in the IWDG_KR register
            ::LL_IWDG_EnableWriteAccess(IWDG);
            // Write the IWDG prescaler by programming IWDG_PR from 0 to 7
            // LL_IWDG_PRESCALER_4 (0) is lowest divider
            ::LL_IWDG_SetPrescaler(IWDG, static_cast<uint32_t>(prescaler));
            // Write the reload register (IWDG_RLR)
            ::LL_IWDG_SetReloadCounter(IWDG, reload);

            // Update registers
            while (::LL_IWDG_IsReady(IWDG) != 1) {
                // Wait for the registers to be updated (IWDG_SR = 0x0000 0000)
            }

            // Refresh the counter value with IWDG_RLR (IWDG_KR = 0x0000 AAAA)
            ::LL_IWDG_ReloadCounter(IWDG);
        }
    }

    //-----------------------------
    // @brief Reload the watchdog.
    void Watchdog::reload() const {
        // Refresh the counter value with IWDG_RLR (IWDG_KR = 0x0000 AAAA)
        ::LL_IWDG_ReloadCounter(IWDG);
    }

    //-----------------------------
    // @brief Validate the watchdog timeout.
    // @param timeout - The timeout for the watchdog.
    // @return bool - True if the timeout is accepted.
    bool Watchdog::is_wdg_timeout(const uint32_t timeout) const {
        // Minimal timeout in microseconds
        const uint32_t min_timeout = 125;
        // Maximal timeout in microseconds
        const uint32_t max_timeout = (8000 * IWDG_RLR_RL);

        // Return if the timeout is valid
        return ((timeout >= min_timeout) && (timeout <= max_timeout));
    }
} // namespace watchdog
